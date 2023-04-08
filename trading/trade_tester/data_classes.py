from pydantic import BaseModel
from enum import IntEnum
from typing import Optional
from datetime import datetime

class Actions(IntEnum):
    Sell = 0
    Buy = 1
    Pass = 2

    def __str__(self) -> str:
        return f'{"Buy" if self == 1 else "Sell"}'


class Order(BaseModel):
    id: int
    type: Actions
    open_time: datetime
    close_time: Optional[datetime] = None
    open: float
    vol: float
    close: float = 0
    TP: float = 0
    SL: float = 0
    pnl: float = 0
    pnl_percent: float = 0

    def __str__(self) -> str:
        return f'Order id {self.id}, type {self.type}, open {self.open}, close {self.close}, pnl {self.pnl}'