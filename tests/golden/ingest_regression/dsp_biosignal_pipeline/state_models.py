from pydantic import BaseModel, Field


class ECGProcessorState(BaseModel):
    filtered: list[float] = Field(default_factory=list)
    rpeaks: list[int] = Field(default_factory=list)

