from pathlib import Path

import dvc.api
import pandas as pd


def download_data() -> pd.DataFrame:
    data_path = Path("industrial_failure/data/equipment_anomaly_data.csv")

    if data_path.exists():
        return pd.read_csv(data_path)

    try:
        with dvc.api.open(
            path=str(data_path),
            repo=".",
            mode="r",
        ) as file:
            return pd.read_csv(file)

    except FileNotFoundError as exc:
        raise FileNotFoundError("Файл не найден. Выполните `dvc pull`.") from exc
