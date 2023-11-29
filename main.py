import pandas as pd

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from src import services
from src.schemas import Item, Items

app = FastAPI()


@app.post("/predict_item")
async def predict_item_route(item: Item) -> float:
    """Prediction of the price of one car based on the passed JSON body.
    Returns selling price value"""
    df = services.prepare_df_from_item(item)
    prediction = services.predict_selling_price(df)
    return prediction[0]


@app.post("/predict_items")
async def predict_items_route(items: Items) -> list[float]:
    """Predicting the price of a list of cars whose parameters are transferred
    in the JSON body. Returns a list of prices"""
    df = services.prepare_df_from_items(items)
    return services.predict_selling_price(df)


@app.post("/predict_items/csv")
async def predict_items_from_csv_route(csv_file: UploadFile = File(...)) -> StreamingResponse:
    """Prediction of the price of cars whose parameters are transferred in
    the csv file. Returns the original file with the added column `selling_price`
    """
    df_raw = pd.read_csv(csv_file.file, index_col=0)
    df_prepared = services.update_input_df_for_prediction(df_raw)
    prediction = services.predict_selling_price(df_prepared)
    return services.prepare_prediction_file_response(df_raw, prediction)
