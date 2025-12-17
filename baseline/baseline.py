#!/usr/bin/env python3
"""
Simple ConvTasNet baseline:
- Input: mono or stereo mixture WAV.
- Output: separated WAVs from pretrained ConvTasNet_WHAM_sepclean.
- Prints: channel type (mono/stereo) and processing time.
"""

import time
import numpy as np
import torch
import torchaudio
import soundfile as sf
from asteroid.models import ConvTasNet  # Pretrained ConvTasNet implementation [web:10]

def convtasnet_baseline(mixture_path,
                        model,
                        target_sr=8000,
                        out_prefix="baseline_sep"):
    # 1) Load mixture with soundfile (same style as your hybrid code) [web:12]
    audio_np, sr = sf.read(mixture_path)  # (T,) or (T, C)

    # 2) Report mono / stereo
    if audio_np.ndim == 1:
        print("Input is MONO")
    else:
        print(f"Input is STEREO with {audio_np.shape[1]} channels")
        # Downmix to mono for ConvTasNet
        audio_np = audio_np.mean(axis=1)

    # 3) Resample to target_sr
    mix_t = torch.from_numpy(audio_np).float().unsqueeze(0)  # (1, T)
    resampler = torchaudio.transforms.Resample(sr, target_sr)
    mix_rs = resampler(mix_t)                               # (1, T_resampled)

    # ConvTasNet expects (batch, 1, time) [web:2][web:6]
    mix_in = mix_rs.unsqueeze(1)                            # (1, 1, T_resampled)

    # 4) Forward pass + timing
    model.eval()
    with torch.no_grad():
        start = time.time()
        est_sources = model(mix_in)                         # (1, n_src, T_resampled)
        proc_time = time.time() - start

    print(f"Processing time: {proc_time:.3f} seconds")

    # 5) Save separated signals
    est_np = est_sources.squeeze(0).cpu().numpy()           # (n_src, T_resampled)
    n_src = est_np.shape[0]
    for i in range(n_src):
        out_path = f"{out_prefix}_{i}.wav"
        sf.write(out_path, est_np[i], target_sr)
        print(f"Saved separated source {i} to {out_path}")

    return proc_time

if __name__ == "__main__":
    mixture_path = "speech_recordings/output_010.wav"  # same style as your hybrid script

    # Load the same pretrained model you are already using [web:1][web:3][web:9]
    model = ConvTasNet.from_pretrained("mpariente/ConvTasNet_WHAM_sepclean")
    model.eval()

    convtasnet_baseline(mixture_path, model)
