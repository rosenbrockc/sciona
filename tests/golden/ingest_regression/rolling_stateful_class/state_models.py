from pydantic import BaseModel, Field


class RollingAveragerState(BaseModel):
    buffer: list[float] = Field(default_factory=list)
    count: int = 0

