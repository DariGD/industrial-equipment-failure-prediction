import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["temperature", "pressure", "vibration", "humidity"]
CATEGORICAL_FEATURES = ["equipment", "location"]


def preprocess_features(df: pd.DataFrame) -> tuple[np.ndarray, ColumnTransformer]:

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    features = preprocessor.fit_transform(df)
    return features, preprocessor
