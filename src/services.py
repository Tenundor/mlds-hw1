from io import BytesIO
from pickle import load

import numpy as np
import pandas as pd

from fastapi.responses import StreamingResponse

from src.schemas import Item, Items


with open("inference_objects.pickle", "rb") as file:
    inference_objects = load(file)


def _recalculate_mileage(row):
    fuel_density = 0.78  # kg/l - average between diesel and gasoline
    if not isinstance(row, str):
        raise TypeError("Mileage value type is not string")
    value, unit = row.split()
    if unit.lower() == "kmpl":
        return float(value)
    elif unit.lower() == "km/kg":
        return float(value) * fuel_density
    else:
        raise ValueError("Invalid mileage value")


def _convert_string_to_float(row):
    value, unit = row.split()
    return float(value)


def update_df_columns_with_units(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_out["mileage"] = df_out["mileage"].apply(_recalculate_mileage)
    for c_name in ("engine", "max_power"):
        df_out[c_name] = df_out[c_name].apply(_convert_string_to_float)
    return df_out


def update_input_df_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    if "selling_price" in df:
        df = df.drop(columns=["selling_price"], axis=1)
    df = df.drop(columns=["name", "torque"], axis=1)
    df = update_df_columns_with_units(df)

    one_hot_mask = df.select_dtypes(include=["object"]).columns.union(["seats"])
    df_to_one_hot = df[one_hot_mask]

    df_numeric = df.drop(columns=one_hot_mask)
    one_hot_encoder = inference_objects["one_hot_encoder"]
    df_dum = one_hot_encoder.transform(df_to_one_hot)
    df = pd.concat([df_dum, df_numeric], axis=1)

    # добавим признак - квадрат года
    df["year_squared"] = df["year"] ** 2

    # добавим признак - мощность двигателя на литр объема
    df["max_power_per_engine_volume"] = df["max_power"] / df["engine"] * 1000

    scaler = inference_objects["scaler"]
    df = pd.DataFrame(data=scaler.transform(df), columns=df.columns)
    return df


def prepare_df_from_item(item: Item) -> pd.DataFrame:
    df = pd.DataFrame([item.model_dump()])
    return update_input_df_for_prediction(df)


def prepare_df_from_items(items: Items) -> pd.DataFrame:
    df = pd.DataFrame([item.model_dump() for item in items.objects])
    return update_input_df_for_prediction(df)


def predict_selling_price(df: pd.DataFrame) -> np.ndarray:
    model = inference_objects["model"]
    prediction_logarithmic = model.predict(df)
    return np.round(np.exp(prediction_logarithmic), 2)


def prepare_prediction_file_response(
        input_df: pd.DataFrame,
        prediction: np.ndarray
) -> StreamingResponse:
    df_out = pd.concat(
        [input_df, pd.DataFrame(prediction, columns=["selling_price"])], axis=1
    )
    output = BytesIO()
    df_out.to_csv(output)
    output.seek(0)
    response = StreamingResponse(
        output,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="prediction.csv"'
        }
    )
    return response
