from pathlib import Path

import hydra
import joblib
import mlflow
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

from industrial_failure.data.download import download_data
from industrial_failure.models.baseline import get_baseline_model
from industrial_failure.utils.metrics import compute_metrics
from industrial_failure.utils.preprocessing import preprocess_features


def train_baseline(cfg: DictConfig) -> None:

    mlflow.set_tracking_uri(cfg.logging.mlflow_uri)
    mlflow.set_experiment(cfg.logging.experiment_name)

    df = download_data()
    features, preprocessor = preprocess_features(df)
    target = df[cfg.data.target_col].values

    skf = StratifiedKFold(
        n_splits=cfg.training.n_splits,
        shuffle=True,
        random_state=cfg.training.random_state,
    )

    metrics_list: list[dict[str, float]] = []

    model_dir = Path("industrial_failure/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params(cfg.model)

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(features, target), start=1
        ):
            train_features, val_features = features[train_idx], features[val_idx]
            train_target, val_target = target[train_idx], target[val_idx]

            model = get_baseline_model(cfg)
            model.fit(train_features, train_target)

            predictions = model.predict(val_features)
            probabilities = model.predict_proba(val_features)[:, 1]

            fold_metrics = compute_metrics(
                val_target,
                predictions,
                probabilities,
            )
            metrics_list.append(fold_metrics)

        final_metrics = {
            metric: float(np.mean([m[metric] for m in metrics_list]))
            for metric in metrics_list[0]
        }

        print("Baseline metrics:", final_metrics)
        mlflow.log_metrics(final_metrics)

        joblib.dump(model, model_dir / "baseline_model.pkl")
        joblib.dump(preprocessor, model_dir / "preprocessor.pkl")


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="baseline",
)
def main(cfg: DictConfig) -> None:
    train_baseline(cfg)


if __name__ == "__main__":
    main()
