# Future Catalog Additions

Gaps identified from end-to-end use cases. These are atoms not yet in
`ageo-atoms` that would extend coverage for real-world pipelines.

## ECG / Cardiac Signal Processing

- **HRV metrics** — SDNN, RMSSD, pNN50, frequency-domain HRV (LF/HF ratio).
  Clinical follow-on after basic heart rate computation.
- **Wavelet transforms** — CWT (continuous wavelet transform) for multi-scale
  QRS detection and feature extraction. Currently only FFT available.
- **Artifact detection** — Motion artifact classification, electrode noise
  detection beyond bandpass filtering. No ICA or noise subspace decomposition.
- **Beat morphology** — P-wave, T-wave, QT interval classification. Individual
  beat segmentation beyond R-peak detection.
- **Multi-lead ECG** — Axis calculation, lead-specific morphology analysis,
  vectorcardiogram support.
- **Detrending** — Hodrick-Prescott filter, polynomial baseline removal for
  long-duration recordings.
