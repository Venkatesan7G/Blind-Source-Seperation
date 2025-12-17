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
    model = ConvTasNet.from_pretrained("mpariente/ConvTasNet_WHAM_sepdur15s")
    model.eval()
    hybrid_bss(mixture_path, model)  # Drag-and-drop: Pass any mono BSS model here





























# #!/usr/bin/env python3
# """
# Blind ConvTasNet separation + Evaluation metrics.
# Outputs: separated_0.wav, separated_1.wav + SIR/SAR/Std.Dev stats
# """

# import time
# import numpy as np
# import torch
# import soundfile as sf
# from asteroid.models import ConvTasNet

# def load_and_resample(path, target_sr=8000):
#     """Load audio using soundfile directly."""
#     audio_np, sr = sf.read(path, dtype="float32")
#     audio = torch.from_numpy(audio_np)
    
#     if len(audio.shape) == 1:
#         audio = audio.unsqueeze(0)
#     else:
#         audio = audio.t()

#     if audio.shape[0] > 1:
#         audio = audio.mean(dim=0, keepdim=True)

#     import torchaudio.transforms as T
#     if sr != target_sr:
#         resampler = T.Resample(sr, target_sr)
#         audio = resampler(audio)

#     return audio, target_sr

# def compute_sir(estimated, reference):
#     """Signal-to-Interference Ratio (dB). Higher = better separation."""
#     # Align lengths
#     min_len = min(estimated.shape[-1], reference.shape[-1])
#     estimated = estimated[..., :min_len]
#     reference = reference[..., :min_len]
    
#     # Remove mean
#     estimated = estimated - estimated.mean()
#     reference = reference - reference.mean()
    
#     # SIR = 10 * log10( ||s||^2 / ||s - s_hat||^2 )
#     signal_power = torch.norm(reference)**2
#     interference_power = torch.norm(reference - estimated)**2
#     sir = 10 * torch.log10(signal_power / (interference_power + 1e-8))
#     return sir.item()

# def compute_sar(estimated, reference):
#     """Speech Activity Ratio. Measures speech-like content."""
#     # Simple energy-based speech activity (threshold at -30dB)
#     energy_est = torch.mean(estimated**2)
#     energy_ref = torch.mean(reference**2)
#     sar = 10 * torch.log10((energy_est + 1e-8) / (energy_ref + 1e-8))
#     return sar.item()

# def evaluate_separation(original, separated):
#     """Evaluate all separated sources against original (assumes original=target1)."""
#     print("\n" + "="*60)
#     print("EVALUATION METRICS (Original vs Separated Sources)")
#     print("="*60)
    
#     orig_std = torch.std(original).item()
#     orig_mean = torch.mean(original).item()
    
#     print(f"Original Audio - Std.Dev: {orig_std:.4f}, Mean: {orig_mean:.4f}")
    
#     metrics = []
#     for i, sep in enumerate(separated):
#         sir = compute_sir(sep, original)
#         sar = compute_sar(sep, original)
        
#         sep_std = torch.std(sep).item()
#         sep_mean = torch.mean(sep).item()
        
#         print(f"Source {i} - SIR: {sir:.2f}dB, SAR: {sar:.2f}dB")
#         print(f"         Std.Dev: {sep_std:.4f}, Mean: {sep_mean:.4f}")
        
#         metrics.append((sir, sar, sep_std, sep_mean))
    
#     # Mic Setup (assuming single mic)
#     print(f"Mic Setup - Single channel mono microphone @ 8kHz")
    
#     return metrics

# def main():
#     mixture_path = "speech_recordings/speechcn.wav"
#     output_prefix = "separated"

#     # Load original mixture (keep copy for evaluation)
#     mix, sr_mix = load_and_resample(mixture_path)
#     original_mix = mix.clone()  # Keep for evaluation
    
#     print(f"Loaded mixture at {sr_mix} Hz, shape: {mix.shape}")

#     # Load model
#     model_tag = "mpariente/ConvTasNet_WHAM_sepclean"
#     print(f"Loading model: {model_tag}...")
#     model = ConvTasNet.from_pretrained(model_tag)
#     model.eval()

#     # Add batch dim
#     mix = mix.unsqueeze(0)

#     # Separation
#     start_time = time.time()
#     with torch.no_grad():
#         est_sources = model(mix)  # (1, n_src, T)
#     proc_time = time.time() - start_time
#     print(f"Processing time: {proc_time:.3f} seconds")

#     # Convert to list of sources (remove batch dim)
#     separated_sources = [est_sources[0, i] for i in range(est_sources.shape[1])]

#     # Save separated sources
#     for i, sep_tensor in enumerate(separated_sources):
#         sep_np = sep_tensor.cpu().numpy()
#         output_file = f"{output_prefix}_{i}.wav"
#         sf.write(output_file, sep_np, sr_mix)
#         print(f"Saved {output_file}")

#     # EVALUATION: Compare separated sources against original mixture
#     evaluate_separation(original_mix, separated_sources)

#     print("\nBlind source separation + evaluation complete!")

# if __name__ == "__main__":
#     main()






# #!/usr/bin/env python3
# """
# Blind ConvTasNet separation at 8 kHz using soundfile DIRECTLY (No torchaudio I/O).
# Input: mixture_path = "speech_recordings/speechcn.wav"
# Outputs: separated_0.wav and separated_1.wav
# """

# import time
# import numpy as np
# import torch
# import soundfile as sf  # Direct import
# from asteroid.models import ConvTasNet

# def load_and_resample(path, target_sr=8000):
#     """Load audio using soundfile directly, avoiding torchaudio backend crashes."""
#     # 1. Read with soundfile (returns numpy array)
#     try:
#         audio_np, sr = sf.read(path, dtype="float32")
#     except Exception as e:
#         raise RuntimeError(f"Failed to read file with soundfile: {e}")

#     # 2. Convert to PyTorch Tensor
#     # soundfile returns (Time, Channels) or (Time,) for mono
#     # PyTorch expects (Channels, Time)
#     audio = torch.from_numpy(audio_np)
    
#     if len(audio.shape) == 1:
#         # Mono case: (Time) -> (1, Time)
#         audio = audio.unsqueeze(0)
#     else:
#         # Stereo/Multi case: (Time, Channels) -> (Channels, Time)
#         audio = audio.t()

#     # 3. Downmix to mono if needed
#     if audio.shape[0] > 1:
#         audio = audio.mean(dim=0, keepdim=True)

#     # 4. Resample if needed
#     # (We can use torchaudio.transforms only if the import doesn't crash)
#     # If torchaudio.transforms crashes, we would need scipy.signal.resample
#     import torchaudio.transforms as T
#     if sr != target_sr:
#         resampler = T.Resample(sr, target_sr)
#         audio = resampler(audio)

#     return audio, target_sr

# def main():
#     mixture_path = "speech_recordings/speechcn.wav"
#     output_prefix = "separated"

#     # Load and resample
#     try:
#         mix, sr_mix = load_and_resample(mixture_path)
#     except Exception as e:
#         print(f"CRITICAL ERROR: {e}")
#         return

#     print(f"Loaded mixture at {sr_mix} Hz, shape: {mix.shape}")

#     # Load pretrained ConvTasNet
#     model_tag = "mpariente/ConvTasNet_WHAM_sepclean"
#     print(f"Loading model: {model_tag}...")
#     model = ConvTasNet.from_pretrained(model_tag)
#     model.eval()

#     # Add batch dim: (1, 1, T)
#     mix = mix.unsqueeze(0)

#     # Separation
#     start_time = time.time()
#     with torch.no_grad():
#         est_sources = model(mix)
#     proc_time = time.time() - start_time
#     print(f"Processing time: {proc_time:.3f} seconds")

#     # Save separated sources using soundfile directly
#     n_src = est_sources.shape[1]
#     for i in range(n_src):
#         # Get source i, remove batch dim: (Time,)
#         sep_tensor = est_sources[0, i] # Shape (T,)
        
#         # Convert back to numpy for soundfile
#         sep_np = sep_tensor.cpu().numpy()
        
#         output_file = f"{output_prefix}_{i}.wav"
        
#         # Save directly with soundfile
#         sf.write(output_file, sep_np, sr_mix)
#         print(f"Saved {output_file}")

#     print("Blind source separation complete!")

# if __name__ == "__main__":
#     main()






















# # #!/usr/bin/env python3
# # """
# # Single-file ConvTasNet separation + SIR/SAR at 48 kHz.

# # Inputs (you must provide these three files):
# #   mix_stereo.wav  : stereo mixture at 48 kHz
# #   s1_stereo.wav   : clean source 1, stereo, aligned with mix
# #   s2_stereo.wav   : clean source 2, stereo, aligned with mix
# # """

# # import time
# # import numpy as np
# # import torch
# # import soundfile as sf

# # from asteroid.models import ConvTasNet
# # from mir_eval.separation import bss_eval_sources


# # def load_stereo_downmix(path, target_sr=None):
# #     """Load stereo wav, downmix to mono, optionally check/resample."""
# #     audio, sr = sf.read(path)          # audio: (T, 2) or (T,)
# #     if audio.ndim == 1:
# #         mono = audio
# #     else:
# #         mono = audio.mean(axis=1)

# #     mono_t = torch.from_numpy(mono).float().unsqueeze(0)  # (1, T)

# #     if target_sr is not None and sr != target_sr:
# #         # If your model is trained at a different SR, resample here.
# #         import torchaudio
# #         mono_t = torchaudio.functional.resample(
# #             mono_t, orig_freq=sr, new_freq=target_sr
# #         )
# #         sr = target_sr

# #     return mono_t, sr


# # def main():
# #     # User parameters
# #     mixture_path = "speech_recordings/speechcn.wav"
# #     s1_path = "s1_stereo.wav"
# #     s2_path = "s2_stereo.wav"

# #     # Your audio is at 48 kHz
# #     mixture_sr_expected = 48000

# #     # Load mixture and references, downmix to mono
# #     mix, sr_mix = load_stereo_downmix(mixture_path)
# #     s1, sr_s1 = load_stereo_downmix(s1_path)
# #     s2, sr_s2 = load_stereo_downmix(s2_path)

# #     # Sanity checks on sample rates
# #     if not (sr_mix == sr_s1 == sr_s2 == mixture_sr_expected):
# #         raise ValueError(
# #             f"All files must be 48 kHz: mix={sr_mix}, s1={sr_s1}, s2={sr_s2}"
# #         )

# #     # Load pretrained ConvTasNet model.
# #     # Replace this tag with any compatible ConvTasNet model you prefer.
# #     # Many ConvTasNet checkpoints are available on the Hugging Face Hub. [web:3][web:12]
# #     model_tag = "mpariente/ConvTasNet_WHAM!_sepclean"
# #     model = ConvTasNet.from_pretrained(model_tag)
# #     model.eval()

# #     # If the model was trained at a different sample rate (e.g., 8 kHz),
# #     # you can either resample your audio before this script or modify
# #     # load_stereo_downmix to resample to model.sample_rate. [web:1][web:15]

# #     # Make all tensors the same length as mixture
# #     T = mix.shape[-1]
# #     s1 = s1[..., :T]
# #     s2 = s2[..., :T]

# #     # Stack references into shape (1, n_src, T)
# #     refs = torch.stack([s1, s2], dim=1)  # (1, 2, T)

# #     # Separation with timing
# #     start_time = time.time()
# #     with torch.no_grad():
# #         est_sources = model(mix)         # (1, n_src, T_pred)
# #     proc_time = time.time() - start_time

# #     # Trim/pad to the same length
# #     T_pred = est_sources.shape[-1]
# #     T_final = min(T, T_pred)
# #     refs = refs[..., :T_final]
# #     est_sources = est_sources[..., :T_final]

# #     # Convert to numpy for mir_eval: (n_src, T)
# #     refs_np = refs.squeeze(0).cpu().numpy()
# #     ests_np = est_sources.squeeze(0).cpu().numpy()

# #     # Compute BSS Eval metrics: SDR, SIR, SAR for each source. [web:7]
# #     sdr, sir, sar, perm = bss_eval_sources(refs_np, ests_np)

# #     # Aggregate metrics for this single file
# #     metrics = {
# #         "SIR_mean": float(np.mean(sir)),
# #         "SIR_std": float(np.std(sir)),
# #         "SAR_mean": float(np.mean(sar)),
# #         "SAR_std": float(np.std(sar)),
# #         "ProcTime_mean": float(proc_time),
# #         "ProcTime_std": 0.0,  # single run
# #     }

# #     # Pretty-print per-source and aggregate metrics
# #     print("Per-source metrics:")
# #     for i in range(len(sir)):
# #         print(
# #             f"  Source {i}: SIR={sir[i]:.3f} dB, SAR={sar[i]:.3f} dB"
# #         )
# #     print("\nAggregate metrics over this file:")
# #     print(
# #         "  SIR  Mean={SIR_mean:.3f} dB  Std={SIR_std:.3f} dB\n"
# #         "  SAR  Mean={SAR_mean:.3f} dB  Std={SAR_std:.3f} dB\n"
# #         "  Proc Time (s) Mean={ProcTime_mean:.3f}  Std={ProcTime_std:.3f}"
# #         .format(**metrics)
# #     )


# # if __name__ == "__main__":
# #     main()
