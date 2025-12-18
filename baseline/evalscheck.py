import argparse
import os
import numpy as np
from scipy.io import wavfile

def load_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    sr, data = wavfile.read(file_path)
    
    # Determine if mono or stereo
    if data.ndim == 1:
        channels = 1
        audio = data.astype(np.float32)
        print(f"{file_path} -> Mono, Sampling rate: {sr} Hz, Length: {len(audio)} samples")
    else:
        channels = data.shape[1]
        # Average channels to mono for metric computation
        audio = data.mean(axis=1).astype(np.float32)
        print(f"{file_path} -> Stereo ({channels} ch), converted to mono, Sampling rate: {sr} Hz, Length: {len(audio)} samples")
    
    return sr, audio, channels

def proj(e, s):
    """Projection of e onto s"""
    denom = np.dot(s, s)
    if denom == 0:
        return np.zeros_like(s)
    return np.dot(e, s) / denom * s

def norm2(x):
    return np.dot(x, x)

def compute_bss_metrics(sources, estimates):
    """Compute SDR, SIR, SAR using BSS Eval metrics with best permutation"""
    n_sources = len(sources)
    from itertools import permutations
    
    best_perm = None
    best_sdr = -np.inf
    
    # Try all permutations to find best matching
    for perm in permutations(range(n_sources)):
        sdr_sum = 0
        for i in range(n_sources):
            s = sources[i]
            e = estimates[perm[i]]
            st = proj(e, s)
            noise = e - st
            sdr = 10 * np.log10(norm2(st) / (norm2(noise) + 1e-10))
            sdr_sum += sdr
        if sdr_sum > best_sdr:
            best_sdr = sdr_sum
            best_perm = perm
    
    # Compute detailed metrics with best permutation
    sdrs, sirs, sars = [], [], []
    for i in range(n_sources):
        s = sources[i]
        e = estimates[best_perm[i]]
        
        st = proj(e, s)                    # target
        e_interf = sum(proj(e, sources[j]) for j in range(n_sources) if j != i)
        e_artif = e - st - e_interf         # artifacts
        
        sdr = 10 * np.log10(norm2(st) / (norm2(e_interf + e_artif) + 1e-10))
        sir = 10 * np.log10(norm2(st) / (norm2(e_interf) + 1e-10))
        sar = 10 * np.log10(norm2(st + e_interf) / (norm2(e_artif) + 1e-10))  # standard SAR definition
        
        sdrs.append(sdr)
        sirs.append(sir)
        sars.append(sar)
    
    return sdrs, sirs, sars, best_perm

def main():
    parser = argparse.ArgumentParser(description="Evaluate Conv-TasNet separation performance")
    parser.add_argument('--orig1', default='./original/audio1.wav', help='Path to original source 1')
    parser.add_argument('--orig2', default='./original/audio2.wav', help='Path to original source 2')
    parser.add_argument('--mixed', default='./speech_recording/mixed.wav', help='Path to mixed audio')
    parser.add_argument('--sep0', default='./speech_recording/separated_0.wav', help='Separated source 0')
    parser.add_argument('--sep1', default='./speech_recording/separated_1.wav', help='Separated source 1')
    args = parser.parse_args()

    print("Loading audio files...\n")
    
    # Load all files
    sr1, src1, ch1 = load_audio(args.orig1)
    sr2, src2, ch2 = load_audio(args.orig2)
    sr_mix, mix, ch_mix = load_audio(args.mixed)
    sr_sep0, sep0, ch_sep0 = load_audio(args.sep0)
    sr_sep1, sep1, ch_sep1 = load_audio(args.sep1)

    print("\n" + "="*60)
    print("SUMMARY OF AUDIO PROPERTIES")
    print("="*60)
    print(f"Original 1: {sr1} Hz, {'Mono' if ch1 == 1 else 'Stereo'}")
    print(f"Original 2: {sr2} Hz, {'Mono' if ch2 == 1 else 'Stereo'}")
    print(f"Mixed:      {sr_mix} Hz, {'Mono' if ch_mix == 1 else 'Stereo'}")
    print(f"Separated 0: {sr_sep0} Hz, {'Mono' if ch_sep0 == 1 else 'Stereo'}")
    print(f"Separated 1: {sr_sep1} Hz, {'Mono' if ch_sep1 == 1 else 'Stereo'}")

    # Check sampling rate consistency
    all_srs = [sr1, sr2, sr_mix, sr_sep0, sr_sep1]
    if len(set(all_srs)) > 1:
        print("\n⚠️  WARNING: Sampling rates differ! Results may be inaccurate.")
    else:
        print("\n✓ All files have the same sampling rate.")

    # Truncate to shortest length
    min_len = min(len(src1), len(src2), len(mix), len(sep0), len(sep1))
    if min_len < max(len(src1), len(src2), len(mix), len(sep0), len(sep1)):
        print(f"Truncating all audios to {min_len} samples for fair comparison.")
    
    src1 = src1[:min_len]
    src2 = src2[:min_len]
    mix = mix[:min_len]
    sep0 = sep0[:min_len]
    sep1 = sep1[:min_len]

    # Baseline: SDR of mixed vs originals
    print("\n" + "="*60)
    print("BASELINE: Mixed audio performance")
    print("="*60)
    sdr_mix1 = 10 * np.log10(norm2(src1) / (norm2(src1 - mix) + 1e-10))
    sdr_mix2 = 10 * np.log10(norm2(src2) / (norm2(src2 - mix) + 1e-10))
    print(f"SDR (Mixed vs Original 1): {sdr_mix1:.2f} dB")
    print(f"SDR (Mixed vs Original 2): {sdr_mix2:.2f} dB")
    print(f"Average SDR (Mixed):       {(sdr_mix1 + sdr_mix2)/2:.2f} dB")

    # Separation metrics
    sources = [src1, src2]
    estimates = [sep0, sep1]
    
    sdrs, sirs, sars, best_perm = compute_bss_metrics(sources, estimates)

    print("\n" + "="*60)
    print("SEPARATION PERFORMANCE (Conv-TasNet)")
    print("="*60)
    print(f"Best permutation: Separated_0 → Original {best_perm.index(0)+1}, Separated_1 → Original {best_perm.index(1)+1}")
    
    for i in range(2):
        orig_idx = i + 1
        sep_idx = best_perm[i]
        print(f"\nOriginal {orig_idx} matched with separated_{sep_idx}.wav:")
        print(f"  SDR (≈ SNR): {sdrs[i]:.2f} dB")
        print(f"  SIR:         {sirs[i]:.2f} dB")
        print(f"  SAR:         {sars[i]:.2f} dB")

    print(f"\nAverage Results:")
    print(f"  SDR: {np.mean(sdrs):.2f} dB")
    print(f"  SIR: {np.mean(sirs):.2f} dB")
    print(f"  SAR: {np.mean(sars):.2f} dB")

if __name__ == "__main__":
    main()