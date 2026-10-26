"""Model training script with MLflow and DagsHub logging.

NOTE: This script trains the model. Per system guidelines, agents must never
execute this script autonomously. Only the user should invoke this script when ready.
"""

import argparse
import os
import keras
from keras import callbacks
import mlflow
import tensorflow as tf

from src.config import AppConfig
from src.data.generator import RWF2000Sequence
from src.models.convlstm import build_model_from_config
from src.utils.mlflow_logger import setup_mlflow_and_dagshub


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ConvLSTM Video Classifier on RWF-2000"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Override MLflow experiment name",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    config = AppConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.experiment_name:
        config.mlflow.experiment_name = args.experiment_name

    print("==================================================")
    print("  ConvLSTM Video Classifier Training Pipeline")
    print("==================================================")
    print(f"Dataset path: {config.data.dataset_path}")
    print(f"Sequence length (diff frames): {config.data.sequence_length}")
    print(f"Image dimensions: {config.data.image_size}")
    is_two_stream = getattr(config.model, "type", "two_stream") == "two_stream"
    if is_two_stream:
        print(f"Pose Conv Filters: {getattr(config.model, 'conv_filters', 9)} | Change ConvLSTM Filters: {getattr(config.model, 'convlstm_diff_filters', 9)}")
        print(f"Aggregator Filters: {getattr(config.model, 'aggregator_filters', 32)} | Depth Multiplier: {getattr(config.model, 'depth_multiplier', 2)}")
    else:
        print(f"ConvLSTM Filters: {getattr(config.model, 'filters', 9)}")
    print(f"Dense sequence: {config.model.dense_units}")
    print(f"Batch size: {config.data.batch_size} | Epochs: {config.training.epochs}")

    # 1. Setup MLflow & DagsHub
    tracking_uri = setup_mlflow_and_dagshub(
        experiment_name=config.mlflow.experiment_name
    )

    # 2. Prepare Datasets
    train_dir = os.path.join(config.data.dataset_path, config.data.train_split)
    val_dir = os.path.join(config.data.dataset_path, config.data.val_split)

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Train/Val directories not found! Expected:\n  {train_dir}\n  {val_dir}\n"
        )

    print(f"\n[Data] Initializing train and validation sequences (two_stream={is_two_stream})...")
    skeletons_path = getattr(config.data, "skeletons_path", None)
    skeletons_train_dir = os.path.join(skeletons_path, config.data.train_split) if skeletons_path else None
    skeletons_val_dir = os.path.join(skeletons_path, config.data.val_split) if skeletons_path else None

    train_seq = RWF2000Sequence(
        split_dir=train_dir,
        skeletons_dir=skeletons_train_dir,
        sequence_length=config.data.sequence_length,
        target_size=(config.data.image_size[0], config.data.image_size[1]),
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        augment=True,
        two_stream=is_two_stream,
    )
    val_seq = RWF2000Sequence(
        split_dir=val_dir,
        skeletons_dir=skeletons_val_dir,
        sequence_length=config.data.sequence_length,
        target_size=(config.data.image_size[0], config.data.image_size[1]),
        batch_size=config.data.batch_size,
        shuffle=False,
        augment=False,
        two_stream=is_two_stream,
    )

    print(f"[Data] Train samples: {len(train_seq.samples)} ({len(train_seq)} batches)")
    print(f"[Data] Val samples:   {len(val_seq.samples)} ({len(val_seq)} batches)")

    # 3. Build Model
    print(f"\n[Model] Building Model (type: {'TwoStream' if is_two_stream else 'SingleStream'})...")
    if is_two_stream:
        from src.models.two_stream import build_two_stream_from_config
        model = build_two_stream_from_config(config)
    else:
        model = build_model_from_config(config)
    model.summary()

    # 4. Compile Model
    optimizer = keras.optimizers.Adam(learning_rate=config.training.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=config.training.loss,
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    # 5. Callbacks
    ckpt_dir = config.training.checkpoint.dir
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = os.path.join(ckpt_dir, "best_convlstm_model.keras")

    training_callbacks = [
        callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor=config.training.checkpoint.monitor,
            mode=config.training.checkpoint.mode,
            save_best_only=config.training.checkpoint.save_best_only,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor=config.training.early_stopping.monitor,
            patience=config.training.early_stopping.patience,
            mode=config.training.early_stopping.mode,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=config.training.early_stopping.monitor,
            factor=config.training.reduce_lr_on_plateau.factor,
            patience=config.training.reduce_lr_on_plateau.patience,
            min_lr=config.training.reduce_lr_on_plateau.min_lr,
            verbose=1,
        ),
    ]

    # 6. Execute Training within MLflow Run
    with mlflow.start_run(run_name=config.mlflow.run_name):
        # Log Hyperparameters
        model_params = {
            "model_type": getattr(config.model, "type", "two_stream"),
            "dense_units": str(config.model.dense_units),
            "sequence_length": config.data.sequence_length,
            "image_size": str(config.data.image_size),
            "batch_size": config.data.batch_size,
            "learning_rate": config.training.learning_rate,
            "optimizer": config.training.optimizer,
            "dropout_rate": config.model.dropout_rate,
        }
        if is_two_stream:
            model_params.update({
                "conv_filters": getattr(config.model, "conv_filters", 9),
                "convlstm_diff_filters": getattr(config.model, "convlstm_diff_filters", 9),
                "aggregator_filters": getattr(config.model, "aggregator_filters", 32),
                "depth_multiplier": getattr(config.model, "depth_multiplier", 2),
                "gamma": getattr(config.model, "gamma", 0.67),
            })
        else:
            model_params.update({
                "filters": getattr(config.model, "filters", 9),
                "kernel_size": str(getattr(config.model, "kernel_size", (3, 3))),
            })
        mlflow.log_params(model_params)

        class MLflowEpochCallback(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    for k, v in logs.items():
                        mlflow.log_metric(k, float(v), step=epoch)

        training_callbacks.append(MLflowEpochCallback())

        print("\n[Train] Commencing model training...")
        history = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=config.training.epochs,
            callbacks=training_callbacks,
            verbose=1,
        )

        if config.mlflow.log_models and os.path.exists(best_model_path):
            mlflow.log_artifact(best_model_path, artifact_path="model_checkpoints")

        print("\n[Done] Training completed successfully!")
        print(f"[Done] Best model saved at: {best_model_path}")


if __name__ == "__main__":
    train()
