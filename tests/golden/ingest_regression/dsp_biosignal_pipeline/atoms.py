class ECGProcessorAtom:
    def run(self, signal):
        return {"filtered": signal, "rpeaks": [], "heart_rate": []}

