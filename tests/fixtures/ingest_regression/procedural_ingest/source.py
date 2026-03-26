def remove_baseline(signal):
    baseline = sum(signal) / len(signal)
    return [value - baseline for value in signal]


def fold_signal(signal, period):
    bins = int(len(signal) / period)
    trimmed = signal[: bins * int(period)]
    folded = []
    for idx in range(int(period)):
        bucket = [trimmed[start + idx] for start in range(0, len(trimmed), int(period))]
        folded.append(sum(bucket) / len(bucket))
    return folded


def compute_snr(folded):
    peak = max(folded)
    floor = min(folded)
    return peak - floor


raw = [1.0, 0.2, 1.4, -0.4, 0.8, 0.1, 1.2, -0.2]
clean = remove_baseline(raw)
folded = fold_signal(clean, 2.0)
snr = compute_snr(folded)

