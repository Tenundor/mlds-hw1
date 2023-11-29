from io import BytesIO
from pickle import load

import pandas as pd
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


app = FastAPI()

with open("inference_objects.pickle", "rb") as file:
    inference_objects = load(file)


class Item(BaseModel):
    name: str
    year: int
    selling_price: int | None = Field(default=None)
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: list[Item]


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


def prepare_df_from_items(items: list[Item]) -> pd.DataFrame:
    df = pd.DataFrame([item.model_dump() for item in items])
    return update_input_df_for_prediction(df)


@app.post("/predict_item")
async def predict_item(item: Item) -> float:
    df = prepare_df_from_item(item)
    model = inference_objects["model"]
    prediction_logarithmic = model.predict(df)
    prediction = np.exp(prediction_logarithmic)
    return prediction[0]


@app.post("/predict_items")
async def predict_items(items: list[Item]) -> list[float]:
    df = prepare_df_from_items(items)
    model = inference_objects["model"]
    prediction_logarithmic = model.predict(df)
    return np.exp(prediction_logarithmic)


@app.post("/predict_items/csv")
async def predict_items_from_csv(csv_file: UploadFile = File(...)) -> StreamingResponse:
    df_raw = pd.read_csv(csv_file.file, index_col=0)
    df_prepared = update_input_df_for_prediction(df_raw)
    model = inference_objects["model"]
    prediction_logarithmic = model.predict(df_prepared)
    prediction = np.round(np.exp(prediction_logarithmic), 2)
    df_out = pd.concat([df_raw, pd.DataFrame(prediction, columns=["selling_price"])], axis=1)
    output = BytesIO()
    df_out.to_csv(output)
    output.seek(0)
    response = StreamingResponse(
        output,
        headers={
            "Content-Disposition": 'attachment; filename="prediction.csv"'
        }
    )
    return response
