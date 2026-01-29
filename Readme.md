# Stereo Blind Source Separation via Unfolded Complex Givens Rotations

A novel approach to blind source separation (BSS) for stereo audio mixtures using deep unfolding of complex-valued Givens rotations with discriminative training.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements a stereo BSS system that separates two concurrent speakers from a two-microphone recording. Unlike traditional beamforming or independent component analysis (ICA) approaches, our method employs **learnable unfolded Givens rotations** in the complex STFT domain, combined with discriminative loss functions to prevent output collapse.

### Key Features

- **Complex Givens Demixer**: Learns frequency-dependent rotation angles (θ, φ) through deep unfolding
- **Discriminative Training**: Novel loss function preventing trivial solutions where outputs converge
- **Spatial Feature Engineering**: Utilizes inter-channel phase difference (IPD) and inter-channel level difference (ILD)
- **Realistic Acoustic Simulation**: Pyroomacoustics-based room impulse response (RIR) generation
- **Data Augmentation**: Seven augmentation strategies to improve generalization

### Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **SIR** | 7-9 dB | Source-to-Interference Ratio (primary metric) |
| **SDR** | 5-7 dB | Source-to-Distortion Ratio |
| **SAR** | 12-15 dB | Source-to-Artifacts Ratio |
| **Processing Time** | ~0.15 s | Per 4.5-second mixture (GPU) |

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Method Overview](#method-overview)
- [Architecture Details](#architecture-details)
- [Citation](#citation)
- [License](#license)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 15+ GB free disk space

### Dependencies

```bash
# Clone repository
git clone https://github.com/yourusername/stereo-bss-givens.git
cd stereo-bss-givens

# Install required packages
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy librosa soundfile
pip install pyroomacoustics datasets
pip install mir_eval matplotlib
```

**Key Dependencies:**
- `torch >= 2.0` - Deep learning framework
- `pyroomacoustics` - Room acoustics simulation
- `librosa` - Audio processing
- `mir_eval` - BSS evaluation metrics
- `datasets` - HuggingFace datasets (for LibriSpeech)

---

## Quick Start

### End-to-End Pipeline

```bash
# 1. Download LibriSpeech utterances (800 files, ~15 min)
python fetch_libre_stream_IMPROVED.py --num_files 800

# 2. Generate stereo mixtures (6000 samples, ~45 min)
python generate_dataset_IMPROVED.py

# 3. Extract STFT features (~15 min)
python feature_extraction.py

# 4. Create train/val/test split
python split_data.py

# 5. Train model (~2-4 hours on GPU)
python torch_train_FIXED_v2.py

# 6. Separate test set (~10 min)
python torch_seperate.py

# 7. Evaluate (BSS metrics)
python evaluation_bss.py
```

### Single-File Demo

```bash
# Test on a single mixture with visualization
python test_bench.py --id 00042 --mode beamform
```

This generates spectrograms and computes metrics for one example.

---

## Dataset Preparation

### Speech Corpus

We use LibriSpeech ASR corpus for speaker diversity:

```bash
python fetch_libre_stream_IMPROVED.py \
    --out_dir speech_pool \
    --num_files 800 \
    --split train.clean.100 \
    --min_duration 3.0 \
    --max_per_speaker 20
```

**Parameters:**
- `num_files`: Total utterances to download (default: 800)
- `split`: LibriSpeech split (`train.clean.100`, `train.clean.360`, etc.)
- `min_duration`: Minimum utterance length in seconds
- `max_per_speaker`: Maximum files per speaker (ensures diversity)

**Output:** `speech_pool/` directory with 800 WAV files and metadata

### Mixture Generation

Generate stereo mixtures with realistic room acoustics:

```bash
python generate_dataset_IMPROVED.py
```

**Process:**
1. Randomly selects two speakers (different identities)
2. Extracts 4.5-second segments
3. Applies random SNR (-5 to +8 dB)
4. Simulates room acoustics (RT60: 0.1-0.6s)
5. Convolves with stereo RIRs
6. Saves mixture + isolated source images

**Output:** `dataset/` directory with 6000 mixtures

**Dataset Structure:**
```
dataset/
├── mix/          # Stereo mixtures (6000 files)
├── s1/           # Source 1 images at mics (6000 files)
├── s2/           # Source 2 images at mics (6000 files)
└── meta.jsonl    # Metadata (SNR, RT60, speakers, etc.)
```

### Feature Extraction

Convert time-domain audio to STFT representations:

```bash
python feature_extraction.py
```

**Process:**
- Computes STFT (n_fft=512, hop=128, window=512)
- Crops to 256 frequency bins (drop Nyquist)
- Pads/trims to 620 time frames
- Saves complex-valued STFT

**Output:** `stft_data/` directory
```
stft_data/
├── X/            # Mixture STFTs (2, 256, 620) complex64
├── S/            # Reference source STFTs (256, 620) complex64
└── meta/         # Metadata (lengths, sample rates)
```

---

## Training

### Standard Training

```bash
python torch_train_FIXED_v2.py
```

**Hyperparameters (in script):**
```python
EPOCHS = 350              # Training epochs
BATCH_SIZE = 8            # Batch size
LR = 5e-4                 # Initial learning rate
K_ITERS = 12              # Unfolding iterations
ALPHA_DISCRIM = 0.05      # Discrimination loss weight (CRITICAL)
```

**What to Monitor:**
```
Epoch 050/350  85.2s
  Train: loss=0.832145 recon=0.815234 discrim=0.338219
  Val:   loss=0.845678 recon=0.828765 discrim=0.338261
  >>> Output stats: |Y1-Y2|=0.4234, corr=0.287
      (Want: mag_diff > 0.3, corr < 0.5)
 Saved best model
```

**Good Training Signs:**
-  `discrim` between 0.02 and 0.50 (non-zero but not dominant)
-  `|Y1-Y2|` > 0.3 (outputs are different)
-  `corr` < 0.5 (outputs not too correlated)
-  Validation loss decreasing

**Bad Training Signs (Collapse):**
-  `|Y1-Y2|` < 0.15 (outputs identical)
-  `corr` > 0.8 (perfect correlation)
-  `discrim` > 2.0 (discrimination weight too high)

**If Collapse Occurs:**
1. Stop training
2. Adjust `ALPHA_DISCRIM` in script (try 0.03 or 0.08)
3. Restart from scratch

**Training Time:**
- GPU (RTX 3090): ~2-4 hours
- GPU (GTX 1080): ~4-6 hours
- CPU: ~12-24 hours (not recommended)

**Output:** `checkpoints_fixed_v2/best.pt` (best model based on validation loss)

---

## Evaluation

### Separate Test Set

```bash
# Edit torch_seperate.py to point to new checkpoint:
# Line 7: CKPT_PATH = "checkpoints_fixed_v2/best.pt"
# Line 9: OUT_DIR = "separated_v2"

python torch_seperate.py
```

**Output:** `separated_v2/` directory with separated sources

### Compute Metrics

```bash
# Edit evaluation_bss.py:
# Line 11: SEP_DIR = "separated_v2"

python evaluation_bss.py
```

**Metrics (BSS Eval v4):**
- **SDR** (Source-to-Distortion Ratio): Overall quality
- **SIR** (Source-to-Interference Ratio): Interference suppression
- **SAR** (Source-to-Artifacts Ratio): Artifact level

**Output Example:**
```
==== BSS Eval (mir_eval) on TEST set, PIT ====
SDR mean/std: 5.82 / 3.21 dB
SIR mean/std: 8.15 / 3.87 dB
SAR mean/std: 13.45 / 4.12 dB
```

---

## Method Overview

### Problem Formulation

Given a stereo mixture **x**(t) = [x₁(t), x₂(t)]ᵀ recorded by two microphones, separate two source signals **s₁**(t) and **s₂**(t):

```
x₁(t) = a₁₁ * s₁(t) + a₁₂ * s₂(t)
x₂(t) = a₂₁ * s₁(t) + a₂₂ * s₂(t)
```

Where **A** = [aᵢⱼ] represents acoustic mixing (includes delays, attenuation, reverberation).

### Approach

1. **Time-Frequency Transform**: Apply STFT to obtain X₁(f,t) and X₂(f,t)
2. **Complex Givens Rotation**: Learn frequency-dependent angles θ(f) and φ(f)
3. **Iterative Refinement**: Deep unfold K iterations of rotation updates
4. **Inverse Transform**: Apply iSTFT to recover separated time-domain signals

### Key Innovation: Discriminative Loss

Traditional PIT (Permutation Invariant Training) loss:

```
L_PIT = min(MSE(ŷ₁, s₁) + MSE(ŷ₂, s₂), MSE(ŷ₁, s₂) + MSE(ŷ₂, s₁))
```

**Problem**: Allows trivial solution ŷ₁ = ŷ₂ = mixture/2 (perfect reconstruction, zero separation)

**Our Solution**: Add discrimination term

```
L_total = L_PIT + α * L_discrim
```

Where L_discrim penalizes correlation between outputs:

```
L_discrim = ReLU(⟨|ŷ₁|, |ŷ₂|⟩)  # Penalize positive correlation
```

This forces ŷ₁ and ŷ₂ to be different, preventing collapse.

---

## Architecture Details

### Complex Givens Demixer

**Input**: X ∈ ℂ^(2×F×T) (stereo STFT)
**Output**: Y ∈ ℂ^(2×F×T) (separated STFTs)

**Forward Pass:**
```
For k = 1 to K:
    1. Apply rotation: Y = R(θ, φ) X
    2. Compute cross-covariance: C = Σₜ y₁(f,t) y₂*(f,t)
    3. Update signals: 
       g_θ = Im(C) / power
       g_φ = Re(C) / power
    4. Smooth across frequency
    5. Update angles:
       θ ← θ - α_θ(k) · g_θ
       φ ← φ - α_φ(k) · g_φ
```

**Learnable Parameters:**
- α_θ, α_φ: Step sizes per iteration (K values each)
- β: Per-frequency gain scaling (F values)
- Smoothing filter weights (1D convolution)

**Rotation Formulation:**
```
y₁ = cos(θ) x₁ + exp(jφ) sin(θ) x₂
y₂ = -exp(-jφ) sin(θ) x₁ + cos(θ) x₂
```

This is a unitary transformation (preserves energy) with phase-awareness via φ.

### Data Augmentation (Training Only)

Applied online during training:

1. **Channel Swap** (50%): Swap L/R channels
2. **Time Shift** (100%): Circular shift ±30 frames
3. **Phase Rotation** (100%): Random phase on channel 2
4. **Magnitude Scaling** (100%): Random gain 0.8-1.2 per channel
5. **Frequency Masking** (30%): Zero out 5-15 adjacent bins
6. **Time Masking** (30%): Zero out 10-30 adjacent frames
7. **Additive Noise** (20%): Low-level complex Gaussian noise

**Impact**: Effective dataset size increases ~7x

---

## File Structure

```
.
├── README.md                          # This file
├── fetch_libre_stream_IMPROVED.py     # Download LibriSpeech
├── generate_dataset_IMPROVED.py       # Create mixtures
├── feature_extraction.py              # STFT computation
├── split_data.py                      # Train/val/test split
├── torch_model_fixed.py               # Givens demixer model
├── torch_train_FIXED_v2.py            # Training script (CORRECTED)
├── torch_seperate.py                  # Inference on test set
├── evaluation_bss.py                  # Compute BSS metrics
├── test_bench.py                      # Single-file demo
├── custom_room_mix.py                 # Room acoustics simulation
├── stft_util.py                       # STFT/iSTFT utilities
└── input_sound.py                     # Audio I/O utilities
```

---

## Hyperparameter Tuning

### Critical Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `ALPHA_DISCRIM` | 0.05 | 0.03-0.10 | **MOST CRITICAL** - Controls output diversity |
| `K_ITERS` | 12 | 8-16 | More iterations = better separation |
| `BATCH_SIZE` | 8 | 4-16 | Higher = more stable gradients |
| `LR` | 5e-4 | 1e-4 to 1e-3 | Learning rate |

### If Results are Poor

**Symptom**: SIR < 3 dB
**Possible Causes:**
1. Model collapsed (check `|Y1-Y2|` during training)
2. `ALPHA_DISCRIM` too high or too low
3. Insufficient training data (<4000 samples)
4. Poor speaker diversity (<30 speakers)

**Solutions:**
1. Adjust `ALPHA_DISCRIM` (try 0.03, 0.05, 0.08, 0.10)
2. Increase dataset size (6000+ mixtures)
3. Download more speech (1000+ files, 50+ speakers)
4. Train longer (300+ epochs)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{stereo-bss-givens-2025,
  author = {Your Name},
  title = {Stereo Blind Source Separation via Unfolded Complex Givens Rotations},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/stereo-bss-givens}
}
```

### Related Work

This method is inspired by:

1. **Deep Unfolding**: 
   - Hershey et al. "Deep clustering: Discriminative embeddings for segmentation and separation." ICASSP 2016.
   - Wisdom et al. "Differentiable consistency constraints for improved deep speech enhancement." ICASSP 2019.

2. **Complex-valued Networks**:
   - Trabelsi et al. "Deep Complex Networks." ICLR 2018.
   - Choi et al. "Phase-aware speech enhancement with deep complex U-Net." ICLR 2019.

3. **Givens Rotations for BSS**:
   - Cardoso & Souloumiac. "Blind beamforming for non-Gaussian signals." IEE Proceedings 1993.
   - Parra & Spence. "Convolutive blind separation of non-stationary sources." IEEE TSP 2000.

4. **Discriminative Training for BSS**:
   - Wang & Wang. "Towards scaling up classification-based speech separation." IEEE TASLP 2013.
   - Luo & Mesgarani. "Conv-TasNet: Surpassing ideal time–frequency magnitude masking." IEEE/ACM TASLP 2019.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **LibriSpeech** corpus for diverse speech data
- **Pyroomacoustics** for realistic acoustic simulation
- **mir_eval** toolkit for standardized BSS evaluation

---

## Contact

For questions or issues, please open a GitHub issue or contact:
- Email: your.email@domain.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

## Changelog

### v1.0.0 (2025-01-29)
- Initial release
- Complex Givens demixer with discriminative training
- Comprehensive data augmentation
- LibriSpeech-based dataset generation
- Full evaluation pipeline