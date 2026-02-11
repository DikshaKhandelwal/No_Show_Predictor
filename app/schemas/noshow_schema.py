from pydantic import BaseModel
from datetime import datetime


class NoShowRequest(BaseModel):
    age: int
    gender: str
    scholarship: int
    diabetes: int
    hipertension: int
    sms_received: int
    handicap: int
    scheduled_day: datetime
    appointment_day: datetime


class NoShowResponse(BaseModel):
    no_show_probability: float
