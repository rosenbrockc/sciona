class RollingAverager:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffer: list[float] = []
        self.count = 0
        self.result = 0.0

    def add_sample(self, value: float) -> None:
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size :]
        self.count += 1

    def compute_average(self) -> float:
        if not self.buffer:
            self.result = 0.0
        else:
            self.result = sum(self.buffer) / len(self.buffer)
        return self.result

