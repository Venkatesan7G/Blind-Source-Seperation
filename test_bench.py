import os
import time
import argparse
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

import torch
from mir_eval.separation import bss_eval_sources

from torch_model import TFMaskUNet
from stft_util import stft_stereo, crop_F, pad_or_trim_T, istft_1d, N_BINS


# Must match training setup
TARGET_F = 256
TARGET_T = 620
EPS = 1e-8
SR = 16000


def load_stereo(path):
    x, sr = sf.read(path, always_2d=True)
    if sr != SR:
        # resample each channel
        x0 = librosa.resample(x[:, 0].astype(np.float32), orig_sr=sr, target_sr=SR)
        x1 = librosa.resample(x[:, 1].astype(np.float32), orig_sr=sr, target_sr=SR)
        x = np.stack([x0, x1], axis=-1)
        sr = SR
    return x.astype(np.float32), sr


def load_ref_mic1(path):
    x, sr = load_stereo(path)
    return x[:, 0].astype(np.float32)


def compute_feats_from_mix(mix_stereo):
    """
    mix_stereo: (N,2) float32
    Returns:
      feats: (F,T,5) float32 with F=256, T=620
      Xc: (2,F,T) complex (cropped/padded)
      orig_len: original sample length
    """
    orig_len = mix_stereo.shape[0]

    # Complex STFT (2,F_full,T_var)
    X = stft_stereo(mix_stereo)  # (2,257,T_var) complex

    # Crop to 256 bins and pad/trim time to 620
    Xc = crop_F(X, TARGET_F)
    Xc = pad_or_trim_T(Xc, TARGET_T)

    X1 = Xc[0]  # (F,T)
    X2 = Xc[1]

    mag1 = np.abs(X1).astype(np.float32)
    mag2 = np.abs(X2).astype(np.float32)

    ipd = (np.angle(X1) - np.angle(X2)).astype(np.float32)
    ild = (np.log(mag1 + EPS) - np.log(mag2 + EPS)).astype(np.float32)

    feats = np.stack([mag1, mag2, np.cos(ipd), np.sin(ipd), ild], axis=-1).astype(np.float32)  # (F,T,5)
    return feats, Xc, orig_len


def separate_one(mix_path, ckpt_path, out_dir="bench_out"):
    os.makedirs(out_dir, exist_ok=True)

    # Load mixture
    mix_stereo, _ = load_stereo(mix_path)

    # Build features
    feats, Xc, orig_len = compute_feats_from_mix(mix_stereo)

    # Load model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    in_ch = ckpt["in_ch"]
    base = ckpt["base"]

    model = TFMaskUNet(in_ch=in_ch, base=base, n_src=2).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Inference + reconstruction timing
    t0 = time.perf_counter()

    x_t = torch.from_numpy(feats).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,5,F,T)
    with torch.no_grad():
        pred_masks = model(x_t)[0].detach().cpu().numpy().astype(np.float32)  # (2,F,T)

    M1 = pred_masks[0]
    M2 = pred_masks[1]

    # Reconstruct using mic1 only (mask-only)
    X1 = Xc[0]  # (F,T) complex
    Y1 = (M1 * X1).astype(np.complex64)
    Y2 = (M2 * X1).astype(np.complex64)

    s1_hat = istft_1d(Y1, length=orig_len)
    s2_hat = istft_1d(Y2, length=orig_len)

    # Normalize to avoid clipping
    mx = max(np.max(np.abs(s1_hat)) + 1e-12, np.max(np.abs(s2_hat)) + 1e-12)
    if mx > 0.99:
        s1_hat = 0.99 * s1_hat / mx
        s2_hat = 0.99 * s2_hat / mx

    t1 = time.perf_counter()
    proc_time = t1 - t0

    # Save outputs
    s1_hat_path = os.path.join(out_dir, "s1_hat.wav")
    s2_hat_path = os.path.join(out_dir, "s2_hat.wav")
    sf.write(s1_hat_path, s1_hat, SR, subtype="FLOAT")
    sf.write(s2_hat_path, s2_hat, SR, subtype="FLOAT")

    return proc_time, s1_hat, s2_hat, s1_hat_path, s2_hat_path


def pit_bss_eval(ref1, ref2, est1, est2):
    """
    Permutation-invariant BSS Eval: tries both assignments and picks best mean SDR.
    Returns (sdr, sir, sar) arrays of shape (2,)
    """
    L = min(len(ref1), len(ref2), len(est1), len(est2))
    ref = np.vstack([ref1[:L], ref2[:L]]).astype(np.float32)

    est_a = np.vstack([est1[:L], est2[:L]]).astype(np.float32)
    est_b = np.vstack([est2[:L], est1[:L]]).astype(np.float32)

    sdr_a, sir_a, sar_a, _ = bss_eval_sources(ref, est_a)
    sdr_b, sir_b, sar_b, _ = bss_eval_sources(ref, est_b)

    if float(np.mean(sdr_b)) > float(np.mean(sdr_a)):
        return sdr_b, sir_b, sar_b
    return sdr_a, sir_a, sar_a


def plot_5_spectrograms(mix_mic1, est1, est2, ref1, ref2, out_png):
    """
    Save one figure with 5 spectrograms.
    """
    signals = [
        ("Mixture (mic1)", mix_mic1),
        ("Estimated s1", est1),
        ("Estimated s2", est2),
        ("Original s1 (mic1)", ref1),
        ("Original s2 (mic1)", ref2),
    ]

    plt.figure(figsize=(14, 12))

    for i, (title, y) in enumerate(signals, start=1):
        # STFT for visualization (not same as training necessarily)
        S = librosa.stft(y.astype(np.float32), n_fft=512, hop_length=128, win_length=512, center=True)
        SdB = librosa.amplitude_to_db(np.abs(S) + 1e-10, ref=np.max)

        ax = plt.subplot(5, 1, i)
        img = librosa.display.specshow(
            SdB,
            sr=SR,
            hop_length=128,
            x_axis="time",
            y_axis="hz",
            ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel("")
        plt.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="00000", help="Dataset id like 00000")
    ap.add_argument("--dataset_dir", default="dataset")
    ap.add_argument("--ckpt", default="checkpoints_torch/best.pt")
    ap.add_argument("--out_dir", default="bench_out")
    args = ap.parse_args()

    sid = args.id
    mix_path = os.path.join(args.dataset_dir, "mix", f"mix_{sid}.wav")
    s1_path = os.path.join(args.dataset_dir, "s1", f"s1_{sid}.wav")
    s2_path = os.path.join(args.dataset_dir, "s2", f"s2_{sid}.wav")

    if not os.path.exists(mix_path):
        raise FileNotFoundError(f"Missing mixture: {mix_path}")
    if not os.path.exists(s1_path) or not os.path.exists(s2_path):
        raise FileNotFoundError("Missing reference s1/s2 WAVs for this id.")

    # Load references (mic1) and mixture mic1
    mix_stereo, _ = load_stereo(mix_path)
    mix_mic1 = mix_stereo[:, 0].astype(np.float32)
    ref1 = load_ref_mic1(s1_path)
    ref2 = load_ref_mic1(s2_path)

    # Separate
    proc_time, est1, est2, est1_path, est2_path = separate_one(
        mix_path=mix_path,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
    )

    # Metrics (PIT)
    sdr, sir, sar = pit_bss_eval(ref1, ref2, est1, est2)

    # Save spectrogram image
    out_png = os.path.join(args.out_dir, f"spectrograms_{sid}.png")
    try:
        import librosa.display  # needed for specshow
        plot_5_spectrograms(mix_mic1, est1, est2, ref1, ref2, out_png)
    except Exception as e:
        print("Spectrogram plot failed:", repr(e))
        out_png = None

    # Print report
    print("\n==== Single-file BSS Bench ====")
    print("ID:", sid)
    print("Mixture:", mix_path)
    print("Ref s1:", s1_path)
    print("Ref s2:", s2_path)
    print("Est s1:", est1_path)
    print("Est s2:", est2_path)
    print(f"\nProcessing time (separation only): {proc_time:.3f} s")
    print(f"SDR (per src): {sdr[0]:.2f}, {sdr[1]:.2f} dB | mean {np.mean(sdr):.2f}")
    print(f"SIR (per src): {sir[0]:.2f}, {sir[1]:.2f} dB | mean {np.mean(sir):.2f}")
    print(f"SAR (per src): {sar[0]:.2f}, {sar[1]:.2f} dB | mean {np.mean(sar):.2f}")
    if out_png:
        print("Spectrogram figure:", out_png)


if __name__ == "__main__":
    main()
