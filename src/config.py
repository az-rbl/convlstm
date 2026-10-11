"""Configuration loader and schema validation."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class DataConfig:
    dataset_path: str = "RWF-2000"
    train_split: str = "train"
    val_split: str = "val"
    sequence_length: int = 16
    frame_stride: int = 2
    image_size: List[int] = field(default_factory=lambda: [112, 112])
    channels: int = 3
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 4
    classes: Dict[str, int] = field(
        default_factory=lambda: {"NonFight": 0, "Fight": 1}
    )


@dataclass
class ModelConfig:
    name: str = "ConvLSTM_ChangeDetection"
    filters: int = 9
    kernel_size: List[int] = field(default_factory=lambda: [3, 3])
    dense_units: List[int] = field(default_factory=lambda: [128, 16, 1])
    dropout_rate: float = 0.3
    recurrent_dropout: float = 0.1
    use_batch_norm: bool = True


@dataclass
class EarlyStoppingConfig:
    patience: int = 7
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class ReduceLROnPlateauConfig:
    patience: int = 3
    factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class CheckpointConfig:
    save_best_only: bool = True
    monitor: str = "val_accuracy"
    mode: str = "max"
    dir: str = "checkpoints"


@dataclass
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 0.0001
    optimizer: str = "adam"
    loss: str = "binary_crossentropy"
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "AUC", "Precision", "Recall"]
    )
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    reduce_lr_on_plateau: ReduceLROnPlateauConfig = field(
        default_factory=ReduceLROnPlateauConfig
    )
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass
class MLflowConfig:
    experiment_name: str = "rwf2000-convlstm-changedetection"
    run_name: str = "convlstm-depth9-diff3ch"
    log_models: bool = True


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AppConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found at: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        data_dict = raw.get("data", {})
        model_dict = raw.get("model", {})
        train_dict = raw.get("training", {})
        mlflow_dict = raw.get("mlflow", {})

        es_dict = train_dict.get("early_stopping", {})
        rlr_dict = train_dict.get("reduce_lr_on_plateau", {})
        ckpt_dict = train_dict.get("checkpoint", {})

        return cls(
            data=DataConfig(**data_dict),
            model=ModelConfig(**model_dict),
            training=TrainingConfig(
                epochs=train_dict.get("epochs", 30),
                learning_rate=train_dict.get("learning_rate", 0.0001),
                optimizer=train_dict.get("optimizer", "adam"),
                loss=train_dict.get("loss", "binary_crossentropy"),
                metrics=train_dict.get("metrics", ["accuracy", "AUC", "Precision", "Recall"]),
                early_stopping=EarlyStoppingConfig(**es_dict),
                reduce_lr_on_plateau=ReduceLROnPlateauConfig(**rlr_dict),
                checkpoint=CheckpointConfig(**ckpt_dict),
            ),
            mlflow=MLflowConfig(**mlflow_dict),
        )
