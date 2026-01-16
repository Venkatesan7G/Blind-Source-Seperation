import os
import json
import time
import argparse
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

import torch
from mir_eval.separation import bss_eval_sources

from torch_model import TFMaskUNet
from stft_util import stft_stereo, crop_F, pad_or_trim_T, istft_1d


# Must match training setup
TARGET_F = 256
TARGET_T = 620
EPS = 1e-8
SR = 16000


def load_stereo(path):
    x, sr = sf.read(path, always_2d=True)
    if sr != SR:
        x0 = librosa.resample(x[:, 0].astype(np.float32), orig_sr=sr, target_sr=SR)
        x1 = librosa.resample(x[:, 1].astype(np.float32), orig_sr=sr, target_sr=SR)
        x = np.stack([x0, x1], axis=-1)
        sr = SR
    return x.astype(np.float32), sr


def load_ref_mic1(path):
    x, _ = load_stereo(path)
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

    X = stft_stereo(mix_stereo)      # (2,257,T_var) complex
    Xc = crop_F(X, TARGET_F)         # (2,256,T_var)
    Xc = pad_or_trim_T(Xc, TARGET_T) # (2,256,620)

    X1 = Xc[0]
    X2 = Xc[1]

    mag1 = np.abs(X1).astype(np.float32)
    mag2 = np.abs(X2).astype(np.float32)

    ipd = (np.angle(X1) - np.angle(X2)).astype(np.float32)
    ild = (np.log(mag1 + EPS) - np.log(mag2 + EPS)).astype(np.float32)

    feats = np.stack([mag1, mag2, np.cos(ipd), np.sin(ipd), ild], axis=-1).astype(np.float32)  # (F,T,5)
    return feats, Xc, orig_len


# ---------------- Beamforming helpers ----------------

def cov_from_mask_vec(X_ft: np.ndarray, m_t: np.ndarray):
    w = (m_t.astype(np.float64) + 1e-12)
    denom = w.sum()
    Xw = X_ft * w[None, :]
    R = (Xw @ X_ft.conj().T) / denom
    return R.astype(np.complex128)


def principal_eigvec(R: np.ndarray):
    vals, vecs = np.linalg.eigh(R)
    v = vecs[:, np.argmax(vals.real)]
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.astype(np.complex128)


def mvdr_weights(Rn: np.ndarray, v: np.ndarray, diag_load=5e-2):
    C = Rn.shape[0]
    tr = np.trace(Rn).real
    Rn = Rn + diag_load * (tr / C + 1e-12) * np.eye(C, dtype=np.complex128)

    invRn = np.linalg.inv(Rn)
    num = invRn @ v
    denom = (v.conj().T @ invRn @ v) + 1e-12
    return (num / denom).astype(np.complex128)


def apply_mvdr_from_target_mask(X: np.ndarray, Mt: np.ndarray, diag_load=5e-2):
    """
    X:  (C,F,T) complex
    Mt: (F,T) float
    """
    C, F, T = X.shape
    Y = np.zeros((F, T), dtype=np.complex64)

    Mn = 1.0 - Mt
    for f in range(F):
        X_ft = X[:, f, :]
        Rs = cov_from_mask_vec(X_ft, Mt[f, :])
        Rn = cov_from_mask_vec(X_ft, Mn[f, :])
        v = principal_eigvec(Rs)
        w = mvdr_weights(Rn, v, diag_load=diag_load)
        Y[f, :] = (w.conj().T @ X_ft).astype(np.complex64)

    return Y


# ---------------- Separation ----------------

def run_model_masks(feats, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_ch = ckpt["in_ch"]
    base = ckpt["base"]

    model = TFMaskUNet(in_ch=in_ch, base=base, n_src=2).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_t = torch.from_numpy(feats).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,5,F,T)
    with torch.no_grad():
        masks = model(x_t)[0].detach().cpu().numpy().astype(np.float32)       # (2,F,T)
    return masks


def separate_one(mix_path, ckpt_path, out_dir, mode="mask", diag_load=5e-2, sharpen_p=1.0):
    """
    mode: "mask" or "beamform"
    Returns: proc_time, est1, est2, paths...
    """
    os.makedirs(out_dir, exist_ok=True)

    mix_stereo, _ = load_stereo(mix_path)
    feats, Xc, orig_len = compute_feats_from_mix(mix_stereo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()

    masks = run_model_masks(feats, ckpt_path, device)
    M1 = masks[0]
    M2 = masks[1]

    # optional sharpening
    if sharpen_p != 1.0:
        M1 = np.power(np.clip(M1, 0, 1), sharpen_p)
        M2 = np.power(np.clip(M2, 0, 1), sharpen_p)
        Z = M1 + M2 + 1e-8
        M1 /= Z
        M2 /= Z

    # stabilization for beamforming
    if mode == "beamform":
        M1 = np.clip(M1, 0.05, 0.95)
        M2 = np.clip(M2, 0.05, 0.95)
        Z = M1 + M2 + 1e-8
        M1 /= Z
        M2 /= Z

    if mode == "mask":
        # mask-only reconstruction using mic1
        X1 = Xc[0]
        Y1 = (M1 * X1).astype(np.complex64)
        Y2 = (M2 * X1).astype(np.complex64)

    elif mode == "beamform":
        # MVDR beamforming using stereo Xc
        Y1 = apply_mvdr_from_target_mask(Xc, M1, diag_load=diag_load)
        Y2 = apply_mvdr_from_target_mask(Xc, M2, diag_load=diag_load)

    else:
        raise ValueError("mode must be 'mask' or 'beamform'")

    est1 = istft_1d(Y1, length=orig_len)
    est2 = istft_1d(Y2, length=orig_len)

    # normalize to avoid clipping
    mx = max(np.max(np.abs(est1)) + 1e-12, np.max(np.abs(est2)) + 1e-12)
    if mx > 0.99:
        est1 = 0.99 * est1 / mx
        est2 = 0.99 * est2 / mx

    t1 = time.perf_counter()
    proc_time = t1 - t0

    p1 = os.path.join(out_dir, f"s1_hat_{mode}.wav")
    p2 = os.path.join(out_dir, f"s2_hat_{mode}.wav")
    sf.write(p1, est1, SR, subtype="FLOAT")
    sf.write(p2, est2, SR, subtype="FLOAT")

    return proc_time, est1, est2, p1, p2


# ---------------- Metrics + plots ----------------

def pit_bss_eval(ref1, ref2, est1, est2):
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
    signals = [
        ("Mixture (mic1)", mix_mic1),
        ("Estimated s1", est1),
        ("Estimated s2", est2),
        ("Original s1 (mic1)", ref1),
        ("Original s2 (mic1)", ref2),
    ]

    plt.figure(figsize=(14, 12))

    for i, (title, y) in enumerate(signals, start=1):
        S = librosa.stft(y.astype(np.float32), n_fft=512, hop_length=128, win_length=512, center=True)
        SdB = librosa.amplitude_to_db(np.abs(S) + 1e-10, ref=np.max)

        ax = plt.subplot(5, 1, i)
        img = librosa.display.specshow(SdB, sr=SR, hop_length=128, x_axis="time", y_axis="hz", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        plt.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def pick_id(args):
    if args.id is not None:
        return args.id

    # random from split
    split = json.load(open(args.split_path, "r", encoding="utf-8"))
    ids = split[args.split]
    rng = np.random.default_rng(args.seed)
    return str(rng.choice(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=None, help="Dataset id like 00000. If omitted, pick random from split.")
    ap.add_argument("--split_path", default="splits/split.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--dataset_dir", default="dataset")
    ap.add_argument("--ckpt", default="checkpoints_torch/best.pt")
    ap.add_argument("--out_dir", default="bench_out")

    ap.add_argument("--mode", default="beamform", choices=["mask", "beamform"])
    ap.add_argument("--diag_load", type=float, default=5e-2)
    ap.add_argument("--sharpen_p", type=float, default=1.0)

    args = ap.parse_args()

    sid = pick_id(args)

    mix_path = os.path.join(args.dataset_dir, "mix", f"mix_{sid}.wav")
    s1_path = os.path.join(args.dataset_dir, "s1", f"s1_{sid}.wav")
    s2_path = os.path.join(args.dataset_dir, "s2", f"s2_{sid}.wav")

    if not os.path.exists(mix_path):
        raise FileNotFoundError(f"Missing mixture: {mix_path}")
    if not os.path.exists(s1_path) or not os.path.exists(s2_path):
        raise FileNotFoundError("Missing reference s1/s2 WAVs for this id.")

    mix_stereo, _ = load_stereo(mix_path)
    mix_mic1 = mix_stereo[:, 0].astype(np.float32)
    ref1 = load_ref_mic1(s1_path)
    ref2 = load_ref_mic1(s2_path)

    proc_time, est1, est2, p1, p2 = separate_one(
        mix_path=mix_path,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        mode=args.mode,
        diag_load=args.diag_load,
        sharpen_p=args.sharpen_p,
    )

    sdr, sir, sar = pit_bss_eval(ref1, ref2, est1, est2)

    out_png = os.path.join(args.out_dir, f"spectrograms_{sid}_{args.mode}.png")
    plot_5_spectrograms(mix_mic1, est1, est2, ref1, ref2, out_png)

    print("\n==== Single-file BSS Bench ====")
    print("Mode:", args.mode)
    print("ID:", sid)
    print("Mixture:", mix_path)
    print("Ref s1:", s1_path)
    print("Ref s2:", s2_path)
    print("Est s1:", p1)
    print("Est s2:", p2)
    print(f"\nProcessing time: {proc_time:.3f} s")
    print(f"SDR (per src): {sdr[0]:.2f}, {sdr[1]:.2f} dB | mean {np.mean(sdr):.2f}")
    print(f"SIR (per src): {sir[0]:.2f}, {sir[1]:.2f} dB | mean {np.mean(sir):.2f}")
    print(f"SAR (per src): {sar[0]:.2f}, {sar[1]:.2f} dB | mean {np.mean(sar):.2f}")
    print("Spectrogram figure:", out_png)


if __name__ == "__main__":
    main()
