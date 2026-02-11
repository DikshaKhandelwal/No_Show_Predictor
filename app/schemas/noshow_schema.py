from pydantic import BaseModel
from typing import Optional


class NoShowRequest(BaseModel):
    patient_id: int
    scheduled_date: str
    appointment_date: str
    age: Optional[int]


class NoShowResponse(BaseModel):
    patient_id: int
    no_show_probability: float
