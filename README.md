# Video Classification with Deep ConvLSTM & Change Detection

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.21+-orange.svg)](https://tensorflow.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org)
[![DagsHub](https://img.shields.io/badge/DagsHub-Integrated-1565C0.svg)](https://dagshub.com)

A spatiotemporal video classification framework tailored for violence detection on surveillance benchmarks like **RWF-2000**.

---

## 🌟 Architecture Highlights

1. **Change Detection Pipeline**:
   - Rather than passing raw video frames, the network processes signed inter-frame consecutive differences:
     $$\Delta I_t = I_{t+1} - I_t, \quad t \in \{1, \dots, T\}$$
   - All **3 RGB channels** are retained to preserve directional color/intensity changes, removing static backgrounds and highlighting movement.
2. **ConvLSTM Backbone**:
   - Single recurrent `ConvLSTM2D` layer with **9 filters**, outputting shape `(B, H, W, 9)`.
   - Captures spatiotemporal motion features directly from the frame difference sequence.
3. **Classification Head**:
   - Global Average Pooling (GAP) collapsing spatial dimensions: $(B, H, W, 9) \to (B, 9)$.
   - Multi-layer fully connected classifier:
     $$\mathbf{GAP} \longrightarrow \mathbf{Dense(128)} \longrightarrow \mathbf{Dense(16)} \longrightarrow \mathbf{Dense(1)} \text{ (Sigmoid)}$$

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
│   │   ├── change_detection.py# Signed inter-frame difference computation
│   │   ├── video_loader.py    # Video decoding and sampling via OpenCV
│   │   ├── generator.py       # RWF-2000 lazy Keras Sequence & tf.data loaders
│   │   └── download_dataset.py# Downloader for Hugging Face RWF-2000
│   ├── models/
│   │   ├── __init__.py
│   │   └── convlstm.py        # Depth-9 ConvLSTM + 128->16->1 FC model builder
│   ├── utils/
│   │   ├── __init__.py
│   │   └── mlflow_logger.py   # DagsHub & MLflow experiment setup
│   ├── train.py               # Training entrypoint (CLI)
│   ├── evaluate.py            # Evaluation & metrics report entrypoint
│   └── predict.py             # Single video inference entrypoint
└── tests/
    ├── __init__.py
    └── test_model.py          # Architectural & shape unit tests
```

---

## 🚀 Getting Started

### 1. Environment & Setup
The repository comes equipped with an existing virtual environment (`.venv`). Activate it using:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure DagsHub & MLflow
Copy `.env.example` to `.env` and fill in your DagsHub credentials:
```bash
cp .env.example .env
```
Edit `.env`:
```env
DAGSHUB_USER_NAME=<your_dagshub_username>
DAGSHUB_REPO_NAME=<your_dagshub_repo_name>
DAGSHUB_TOKEN=<your_dagshub_token>
```
> *Note:* If `.env` is omitted, MLflow automatically falls back to local tracking in `./mlruns`.

### 3. Obtain the RWF-2000 Dataset
Download and extract the RWF-2000 dataset from Hugging Face:
```bash
python -m src.data.download_dataset --dest RWF-2000
```
Expected folder structure:
```
RWF-2000/
├── train/
│   ├── Fight/
│   └── NonFight/
└── val/
    ├── Fight/
    └── NonFight/
```

### 4. Run Architectural Unit Tests
Verify model shape flow, change detection logic, and 9-layer depth without initiating training:
```bash
pytest tests/test_model.py
```

---

## 🏃 Running the Workflow (User Triggered)

> ⚠️ **Important:** Training is never executed automatically. You must run the command below when you are ready to train the model.

### Train the Model
```bash
python -m src.train --config config/default.yaml
```
CLI override options:
```bash
python -m src.train --epochs 30 --batch_size 8 --lr 0.0001
```

### Evaluate on Validation Split
```bash
python -m src.evaluate --model_path checkpoints/best_convlstm_model.keras
```

### Predict on a Single Video
```bash
python -m src.predict --video path/to/sample.avi --model_path checkpoints/best_convlstm_model.keras
```
