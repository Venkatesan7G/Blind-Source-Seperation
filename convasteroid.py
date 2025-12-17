#!/usr/bin/env python3
"""
Elegant Hybrid BSS: Random Directions (Stereo Priors) + Quantization/Dither + Pretrained ConvTasNet.
- Drag-and-drop: Function takes any mono BSS model.
- Input: Stereo mixture WAV.
- Output: Separated WAVs.
"""

import time
import numpy as np
import torch
import torchaudio

# Force soundfile backend to fix TorchCodec/FFmpeg DLL error on Windows
torchaudio.set_audio_backend("soundfile")

import scipy.signal
import joblib
from asteroid.models import ConvTasNet  # Pretrained example model

# Random Directions Functions (adapted from GitHub for efficiency)
def allp_delayfilt(tau, maxdelay=20):
    L = int(tau) + 1
    n = np.arange(0, L)
    a = np.append([1.0], np.cumprod(((L - n) * (L - n - tau)) / ((n + 1) * (n + 1 + tau))))
    b = np.flipud(a)
    a_padded = np.append(a, np.zeros(maxdelay + 2 - len(a)))
    b_padded = np.append(b, np.zeros(maxdelay + 2 - len(b)))
    return a_padded, b_padded

def abskl_multichan(Xunm, chanout):
    X_abs = np.abs(Xunm)
    for tochan in range(chanout):
        X_abs[:, tochan] /= (np.sum(X_abs[:, tochan]) + 1e-6)
    abskl = 0.0
    for k0 in range(chanout - 1):
        for k1 in range(k0 + 1, chanout):
            abskl += np.sum(X_abs[:, k0] * np.log((X_abs[:, k0] + 1e-6) / (X_abs[:, k1] + 1e-6)))
            abskl += np.sum(X_abs[:, k1] * np.log((X_abs[:, k1] + 1e-6) / (X_abs[:, k0] + 1e-6)))
    return -abskl

def unmixing(coeffs, X, chanout, maxdelay=20):
    chanin = X.shape[1]
    att = np.reshape(coeffs[:chanin * chanout], (chanin, chanout))
    delay = np.abs(np.reshape(coeffs[chanin * chanout:], (chanin, chanout)))
    att = np.clip(att, -1.5, 1.5)
    delay = np.clip(delay, 0, maxdelay)
    Y = np.zeros_like(X[:, :, np.newaxis]) * chanout  # (samples, chanin, chanout)
    for fromchan in range(chanin):
        for tochan in range(chanout):
            a, b = allp_delayfilt(delay[fromchan, tochan], maxdelay)
            Y[:, fromchan, tochan] = scipy.signal.lfilter(b, a, X[:, fromchan])
    X_sep = np.sum(Y * att[np.newaxis, :, :], axis=1)  # (samples, chanout)
    return X_sep

def objfunc(coeffs, args):
    X, chanout = args
    Xunm = unmixing(coeffs, X, chanout)
    inpow = np.sum(X**2) / X.shape[1]
    outpow = np.sum(Xunm**2, axis=0)
    negabskl = abskl_multichan(Xunm, chanout) + 0.1 * np.sum(np.abs(outpow / inpow - 1))
    return negabskl

def optimrandomdir(objfunction, coeffs, args=(), iterations=200, startingscale=2, endscale=0.0):
    sh = coeffs.shape
    X0 = objfunction(coeffs, args)
    parallelset = joblib.cpu_count()
    coeffsmin = coeffs.copy()
    Xmin = X0
    for m in range(0, iterations, parallelset):
        scale = (startingscale - endscale) * ((1.0 - m / iterations) ** 2) + endscale
        scale = max(scale, 1e-4)
        def trial():
            coeffvar = np.random.normal(0, scale, sh)
            c1 = coeffsmin + coeffvar
            return objfunction(c1, args), c1
        results = joblib.Parallel(n_jobs=parallelset)(joblib.delayed(trial)() for _ in range(parallelset))
        X1s = [r[0] for r in results]
        min_idx = np.argmin(X1s)
        X1, c1 = results[min_idx]
        if X1 < Xmin:
            coeffsmin = c1
            Xmin = X1
    return coeffsmin

# Preprocessing: Quantization + Dither
def quantize_dither(waveform, bit_depth=16):
    scale = 2 ** (bit_depth - 1)
    waveform = np.clip(waveform * scale, -scale, scale - 1)
    dither = np.random.uniform(-0.5, 0.5, waveform.shape) / scale
    waveform += dither
    waveform = np.round(waveform) / scale  # Back to float [-1,1]
    return waveform

# Hybrid BSS Wrapper (Drag-and-Drop for Any Model)
def hybrid_bss(mixture_path, bss_model, n_src=2, target_sr=8000, blend_weight=0.3):
    """
    Hybrid Process:
    1. Load stereo mixture, apply quantize/dither.
    2. Run Random Directions on stereo for initial separations (priors).
    3. Downmix to mono, resample.
    4. Blend mono mix with downmixed initials (intelligent input enhancement).
    5. Feed to pretrained BSS model for refinement.
    """
    # Load and preprocess
    audio, sr = torchaudio.load(mixture_path)
    if audio.shape[0] == 1:  # If mono, duplicate for stereo processing
        audio = audio.repeat(2, 1)
    audio_np = audio.numpy().T  # (samples, channels=2)
    audio_np = quantize_dither(audio_np)

    # Random Directions for priors (initial sep)
    chanin, chanout = audio_np.shape[1], n_src
    num_coeffs = chanin * chanout * 2  # att + delays
    initial_coeffs = np.zeros(num_coeffs)
    coeffs_min = optimrandomdir(objfunc, initial_coeffs, args=(audio_np, chanout))
    initial_sep = unmixing(coeffs_min, audio_np, chanout)  # (samples, n_src)

    # Downmix/resample original to mono 8kHz
    mono_mix = audio.mean(dim=0).unsqueeze(0)  # (1, T)
    resampler = torchaudio.transforms.Resample(sr, target_sr)
    mono_mix = resampler(mono_mix).squeeze().numpy()

    # Resample/blend initials with mix (weighted priors)
    initial_sep_t = torch.from_numpy(initial_sep.T).float().unsqueeze(0)  # (1, n_src, T)
    initial_sep_t = resampler(initial_sep_t).squeeze(0).numpy()  # (n_src, T_resampled)
    enhanced_inputs = []  # List for each source
    for i in range(n_src):
        blended = (1 - blend_weight) * mono_mix + blend_weight * initial_sep_t[i]
        enhanced_inputs.append(torch.from_numpy(blended).float().unsqueeze(0).unsqueeze(0))  # (1, 1, T) per source

    # Refine with BSS model (sequential per enhanced input)
    start = time.time()
    with torch.no_grad():
        refined_sources = []
        for enhanced in enhanced_inputs:
            sep = bss_model(enhanced)  # (1, n_src, T); take first as refinement
            refined_sources.append(sep[0, 0].cpu())  # Simplify: Take primary sep per input
    proc_time = time.time() - start

    # Save (stack if needed; here save per source)
    for i, sep in enumerate(refined_sources):
        torchaudio.save(f"separated_{i}.wav", sep.unsqueeze(0), target_sr)

    print(f"Hybrid BSS complete. Time: {proc_time:.3f}s")

if __name__ == "__main__":
    mixture_path = "speech_recordings/speechcn.wav"
    # Load pretrained ConvTasNet (example model)
    model = ConvTasNet.from_pretrained("mpariente/ConvTasNet_WHAM_sepclean")
    model.eval()
    hybrid_bss(mixture_path, model)  # Drag-and-drop: Pass any mono BSS model here