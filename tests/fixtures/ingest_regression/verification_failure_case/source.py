class BrokenEstimator:
    def fit(self, x, y):
        self.model = x + y
        return self

