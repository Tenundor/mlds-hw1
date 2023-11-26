from pickle import load

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

with open("inference_objects.pickle", "rb") as file:
    inference_objects = load(file)


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
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


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    print(f"item: {item}")
    print(inference_objects["model"].coef_)
    return 100500.


@app.post("/predict_items")
def predict_items(items: list[Item]) -> list[float]:
    print(f"items: {items}")
    return [100500., 200500., 300500.]
