# Gravitational Wave Signal Reconstruction

Deep learning pipeline for denoising and reconstructing gravitational wave signals from real LIGO strain data, framed as a time-series-to-time-series reconstruction task. Benchmarks an LSTM baseline against a Transformer architecture on the landmark GW150914 binary black hole merger event.

---

## Results

| Model | SNR (dB) | Overlap Score | MSE |
|---|---|---|---|
| LSTM Baseline | 0.07 | — | higher |
| **Transformer** | **5.7** | higher | lower |

> Transformer achieves **81x better SNR** than the LSTM baseline on the same data.

---

## Dataset

- **Source:** [LIGO Open Science Center (GWOSC)](https://gwosc.org/)
- **Events:** GW150914 (Binary Black Hole merger, first detected GW), GW151226
- **Detectors:** Hanford (H1) and Livingston (L1)
- **Sampling rate:** 4096 Hz
- **Format:** HDF5 strain files

---

## Pipeline

### 1. Preprocessing
Raw LIGO strain data is extremely noisy. The following steps are applied before any model sees the data:

- **Detrending** — remove mean offset
- **Spectral whitening** — flatten noise PSD using Welch's method so the merger signal becomes visible
- **Bandpass filtering** — 4th-order Butterworth filter isolating 30–300 Hz (Binary Black Hole merger frequencies)
- **Normalization** — zero mean, unit variance
- **Merger window extraction** — focused on ±2s around the GW150914 merger at t=16.3s
- **Sliding window sequences** — 2048-timestep windows (0.5s at 4096 Hz) with stride=64 for overlapping samples

### 2. Task Formulation
Framed as **next-step prediction**: given timesteps `[0, N-1]`, predict `[1, N]`. This forces the model to learn the underlying waveform structure rather than memorize noise.

### 3. Models

**LSTM Baseline**
- Stacked LSTM layers for sequential modeling
- Standard MSE loss

**Transformer (improved)**
- Sinusoidal positional encoding (Vaswani et al. 2017)
- Multi-head self-attention blocks (8 heads)
- Feed-forward sublayers with residual connections + LayerNorm
- Custom **physical overlap loss**: `MSE + 10 × (1 − overlap_score)`
  - The overlap score `⟨h₁|h₂⟩ / √(⟨h₁|h₁⟩⟨h₂|h₂⟩)` is the standard match metric in GW astronomy
- Training: Early stopping + ReduceLROnPlateau callbacks

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Signal Processing | SciPy (`butter`, `filtfilt`, `welch`) |
| Data | h5py, NumPy, Pandas |
| Visualization | Matplotlib |
| Platform | Google Colab (T4 GPU) |

---

## Repository Structure

```
├── Gravitational_Wave_Signal_Reconstruction.ipynb   # Full pipeline notebook
├── model_comparison.csv                              # Results table (generated)
├── best_transformer_v2.h5                            # Best transformer weights (generated)
├── best_lstm.h5                                      # Best LSTM weights (generated)
└── README.md
```

---

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Download LIGO strain data from [GWOSC](https://gwosc.org/data/) for GW150914 (H1 and L1, HDF5 format)
3. Upload the `.hdf5` files when prompted by the data loading cell
4. Run all cells sequentially — preprocessing → LSTM → Transformer → comparison

---

## Key References

- Abbott et al. (2016) — *Observation of Gravitational Waves from a Binary Black Hole Merger*, PRL
- Vaswani et al. (2017) — *Attention Is All You Need*
- [LIGO Open Science Center](https://gwosc.org/)

---

*Built as part of a deep learning research project exploring sequence-to-sequence architectures on real astrophysical data.*
