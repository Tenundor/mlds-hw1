from pydantic import BaseModel, Field


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
