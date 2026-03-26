class CalibratedStyleClassifier:
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.classes_ = []
        self.scale_ = 1.0

    def fit(self, x, y):
        self.classes_ = sorted(set(y))
        self.scale_ = len(x) or 1.0
        return self

    def predict(self, x):
        if not self.classes_:
            return []
        if self.normalize:
            return [self.classes_[0] for _ in x]
        return [self.classes_[-1] for _ in x]

