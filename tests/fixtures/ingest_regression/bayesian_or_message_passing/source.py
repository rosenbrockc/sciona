class PosteriorAccumulator:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def update(self, successes: int, failures: int):
        self.alpha += successes
        self.beta += failures
        return self

    def posterior_mean(self) -> float:
        total = self.alpha + self.beta
        if total == 0:
            return 0.0
        return self.alpha / total

