from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_baseline_model(cfg):

    if cfg.model.type == "logistic":
        model = LogisticRegression(
            class_weight=cfg.model.class_weight, max_iter=cfg.model.max_iter
        )
    elif cfg.model.type == "xgboost":
        model = XGBClassifier(
            scale_pos_weight=cfg.model.scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model
