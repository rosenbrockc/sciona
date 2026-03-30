class PeakDetector:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def detect(self, signal):
        peaks = [idx for idx, value in enumerate(signal) if value >= self.threshold]
        quality = 1.0 if peaks else 0.0
        return {
            "rpeaks": peaks,
            "quality": quality,
        }
