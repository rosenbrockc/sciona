def remove_baseline_atom(signal):
    baseline = sum(signal) / len(signal)
    return [value - baseline for value in signal]


def fold_signal_atom(signal, period):
    return signal, period


def compute_snr_atom(folded):
    return max(folded) - min(folded)

