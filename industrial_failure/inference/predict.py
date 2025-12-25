import joblib
import torch
import pandas as pd

from industrial_failure.utils.preprocessing import preprocess_features
from industrial_failure.models.NN import EquipmentNN


def load_baseline_model(model_path: str, preprocessor_path: str):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


def load_nn_model(model_path: str, preprocessor_path: str, input_dim: int, cfg):
    model = EquipmentNN(cfg)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


def predict_baseline(df: pd.DataFrame, model, preprocessor):
    X = preprocessor.transform(df)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return probs, preds


def predict_nn(df: pd.DataFrame, model, preprocessor):
    X = preprocessor.transform(df)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        probs = model(X_tensor).squeeze().numpy()
    preds = (probs >= 0.5).astype(int)
    return probs, preds


def predict_from_csv(file_path: str, model_type="baseline", cfg=None):
    df = pd.read_csv(file_path)

    if model_type == "baseline":
        model, preprocessor = load_baseline_model(
            "industrial_failure/models/baseline_model.pkl",
            "industrial_failure/models/preprocessor.pkl",
        )
        probs, preds = predict_baseline(df, model, preprocessor)

    elif model_type == "nn":
        X_sample = preprocess_features(df)[0]
        input_dim = X_sample.shape[1]
        model, preprocessor = load_nn_model(
            "industrial_failure/models/neural_network.pth",
            "industrial_failure/models/preprocessor_nn.pkl",
            input_dim,
            cfg,
        )
        probs, preds = predict_nn(df, model, preprocessor)
    else:
        raise ValueError("Unknown model_type, choose 'baseline' or 'nn'")

    df["prob_faulty"] = probs
    df["pred_faulty"] = preds
    return df


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../../configs", config_name="nn")
    def main(cfg: DictConfig):
        fire.Fire(
            lambda file_path, model_type="baseline": predict_from_csv(
                file_path, model_type=model_type, cfg=cfg
            )
        )

    main()
