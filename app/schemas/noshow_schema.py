from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class NoShowRequest(BaseModel):
    age: int
    gender: str
    scholorship: int
    diabetes: int
    alcoholism: int
    sms_received: int
    neighbourhood: str
    handicap: int
    scheduled_day: datetime
    appointment_day: datetime


class NoShowResponse(BaseModel):
    no_show_probability: float
