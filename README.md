# Two-Stream Video Classification: ConvLSTM & Pose Estimation

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.21+-orange.svg)](https://tensorflow.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org)
[![DagsHub](https://img.shields.io/badge/DagsHub-Integrated-1565C0.svg)](https://dagshub.com)

A spatiotemporal two-stream deep learning framework replicating the architecture from **Garcia-Cobo & SanMiguel (CVIU 2023)**: *"Human skeletons and change detection for efficient violence detection in surveillance videos"*.

---

## 🌟 Architecture Highlights

The network converges two complementary streams before spatiotemporal aggregation:

```
[Raw Video Stream]                               [Pose / Skeleton Stream]
         │                                                  │
Frame Differencing (ΔI = I_{t+1} - I_t)           Gamma Correction (γ = 0.67)
         │                                                  │
Batch Normalization                               OpenPose (Skeletons on Black Background)
         │                                                  │
ConvLSTM2D (9 filters, 3x3, tanh)                 TimeDistributed Conv2D (9 filters, 3x3, ReLU)
         │                                                  │
         │                                        BatchNorm (scale=False, center=False)
         └───────────────────────┬──────────────────────────┘
                                 │
                   Pointwise Addition Fusion (Eq. 5)
                     F_combined = BatchNorm(F_RGB) ⊕ tanh(F_Change)
                                 │
                   ConvLSTM2D Temporal Aggregator (32 filters, tanh)
                                 │
                   DepthwiseConv2D Post-Processing (depth_multiplier=2 -> 64 filters)
                                 │
                   GlobalAveragePooling2D
                                 │
                   Dense(128, ReLU) -> Dropout
                                 │
                   Dense(16, ReLU) -> Dropout
                                 │
                   Dense(1, Sigmoid) [Fight / NonFight]
```

- **Exact Trainable Parameters**: **62,583** (matching the best performing model in Table 7 of the CVIU 2023 paper).

---

## 📁 Repository Layout

```
convlstm/
├── .gitignore
├── .env.example               # Template for DagsHub / MLflow credentials
├── requirements.txt           # Project dependencies
├── README.md
├── config/
│   ├── default.yaml           # Model, data, and training hyperparameters
│   └── dagshub.yaml          # DagsHub integration settings
├── src/
│   ├── __init__.py
│   ├── config.py              # YAML configuration parser and dataclass
│   ├── data/
│   │   ├── __init__.py
│   │   ├── change_detection.py    # Signed inter-frame differences (ΔI)
│   │   ├── gamma.py               # Fast lookup-table gamma correction (γ = 0.67)
│   │   ├── pose_extractor.py      # Skeleton rendering on black backgrounds
│   │   ├── preprocess_skeletons.py# Batch offline utility for rendering skeleton videos
│   │   ├── video_loader.py        # Video decoding and sampling via OpenCV
│   │   ├── generator.py           # Two-stream lazy Keras Sequence & tf.data loaders
│   │   └── download_dataset.py    # Downloader for Hugging Face RWF-2000
│   ├── models/
│   │   ├── __init__.py
│   │   ├── two_stream.py          # Complete Two-Stream converging architecture
│   │   └── convlstm.py            # Single-stream ConvLSTM model variant
│   ├── utils/
│   │   ├── __init__.py
│   │   └── mlflow_logger.py       # DagsHub & MLflow experiment setup
│   ├── train.py                   # Training entrypoint (CLI)
│   ├── evaluate.py                # Evaluation & metrics report entrypoint
│   └── predict.py                 # Single video inference entrypoint (two-stream aware)
└── tests/
    ├── __init__.py
    ├── test_model.py              # Architectural tests for single-stream ConvLSTM
    └── test_two_stream.py         # Full two-stream converging network tests
```

---

## 🚀 Getting Started

### 1. Environment & Setup
Activate your environment and ensure dependencies are installed:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure DagsHub & MLflow
Copy `.env.example` to `.env` and fill in your credentials (or leave blank to use local MLflow in `./mlruns`):
```bash
cp .env.example .env
```

### 3. Download the Dataset
```bash
python -m src.data.download_dataset --dest data/raw/RWF-2000
```

### 4. Preprocess Skeletons on Black Backgrounds
Run the preprocessing utility to apply gamma correction ($\gamma = 0.67$) and render skeleton videos on black backgrounds:
```bash
python -m src.data.preprocess_skeletons --dataset_dir data/raw/RWF-2000 --output_dir data/processed/RWF-2000 --gamma 0.67 --size 112
```
This produces matching `*_skeletons.avi` files alongside the original videos.

### 5. Run Verification Tests (Zero Model Training)
Verify all components with unit tests using synthetic tensors:
```bash
pytest
# Output: 7 passed
```

---

## 🏃 Running the Workflow (User Triggered)

> ⚠️ **Important:** As an AI pair programmer, I will never initiate model training autonomously. Use the commands below when you are ready to run them yourself:

### Replicate Exact Garcia-Cobo & SanMiguel (CVIU 2023) Experiment
Run the standalone replication script (exact 62,583 trainable parameters, 100x100 resolution, 51 frames, batch size 10, Adam fixed lr 0.001, 50 epochs, seed 0):
```bash
python train_original.py
# Dry run verification without starting training:
python train_original.py --dry_run
```
Or via the modular training pipeline with the paper config:
```bash
python -m src.train --config config/garcia_cviu_2023.yaml
```

### Train with Flexible Default Config
```bash
python -m src.train --config config/default.yaml
```

### Evaluate on Validation Set
```bash
python -m src.evaluate --model_path checkpoints/best_model.h5 --config config/garcia_cviu_2023.yaml
```

### Run Prediction on a Single Video
```bash
python -m src.predict --video path/to/sample.avi --model_path checkpoints/best_model.h5 --config config/garcia_cviu_2023.yaml
```
