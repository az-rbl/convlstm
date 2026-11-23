"""MLflow and DagsHub tracking utilities and callbacks."""

import os
from typing import Optional
from dotenv import load_dotenv
import mlflow
import yaml


def setup_mlflow_and_dagshub(
    experiment_name: str = "rwf2000-convlstm",
    dagshub_yaml_path: str = "config/dagshub.yaml",
) -> str:
    """Configures MLflow tracking with DagsHub integration or local fallback.

    Args:
        experiment_name: Name of MLflow experiment.
        dagshub_yaml_path: Path to dagshub.yaml settings.

    Returns:
        tracking_uri: Active MLflow tracking URI.
    """
    load_dotenv()  # Load .env variables

    repo_owner = os.getenv("DAGSHUB_USER_NAME")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("DAGSHUB_TOKEN")

    # Read dagshub.yaml if env vars are partially set
    if os.path.exists(dagshub_yaml_path):
        with open(dagshub_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            dh_cfg = cfg.get("dagshub", {})
            if not repo_owner:
                repo_owner = dh_cfg.get("repo_owner")
            if not repo_name:
                repo_name = dh_cfg.get("repo_name")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri and repo_owner and repo_name and token:
        try:
            import dagshub

            dagshub.init(
                repo_owner=repo_owner,
                repo_name=repo_name,
                mlflow=True,
            )
            print(f"[DagsHub] Connected to remote repository: {repo_owner}/{repo_name}")
            tracking_uri = mlflow.get_tracking_uri()
        except Exception as e:
            print(f"[DagsHub] Failed to initialize via SDK: {e}. Falling back...")
            tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
            os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            mlflow.set_tracking_uri(tracking_uri)
    elif not tracking_uri:
        # Fallback to local SQLite backend (file store is deprecated in recent MLflow)
        tracking_uri = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[MLflow] Using local SQLite tracking URI: {tracking_uri}")
    else:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[MLflow] Tracking URI set to: {tracking_uri}")

    mlflow.set_experiment(experiment_name)
    return mlflow.get_tracking_uri()
