class ECGProcessor:
    def __init__(self, signal, sampling_rate: float = 1000.0):
        self.signal = signal
        self.sampling_rate = sampling_rate
        self.filtered = []
        self.rpeaks = []
        self.heart_rate = []

    def filter_signal(self):
        self.filtered = [value for value in self.signal if abs(value) > 0.1]

    def detect_rpeaks(self):
        self.rpeaks = [idx for idx, value in enumerate(self.filtered) if value > 0.8]

    def compute_heart_rate(self):
        if len(self.rpeaks) < 2:
            self.heart_rate = []
            return
        rr = [
            float(self.rpeaks[idx + 1] - self.rpeaks[idx]) / self.sampling_rate
            for idx in range(len(self.rpeaks) - 1)
        ]
        self.heart_rate = [60.0 / interval for interval in rr if interval > 0]

