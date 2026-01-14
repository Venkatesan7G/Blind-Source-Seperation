import os
import json
import numpy as np
import soundfile as sf
import torch

from torch_model import TFMaskUNet
from stft_util import istft_1d

STFT_DIR = "stft_data"
SPLIT_PATH = "splits/split.json"
CKPT_PATH = "checkpoints_torch/best.pt"
OUT_DIR = "separated_beamform"

FEATS_DIR = os.path.join(STFT_DIR, "feats")
X_DIR = os.path.join(STFT_DIR, "X")
META_DIR = os.path.join(STFT_DIR, "meta")

SR = 16000
EPS = 1e-8

DIAG_LOAD = 1e-2   # stronger for stability
MASK_SHARPEN_P = 1.0  # try 1.0 first; later 0.7..1.5


def cov_from_mask_vec(X_ft: np.ndarray, m_t: np.ndarray):
    """
    Vectorized covariance.
    X_ft: (C,T) complex
    m_t:  (T,) float mask
    Returns: (C,C) complex covariance
    """
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


def mvdr_weights(Rn: np.ndarray, v: np.ndarray, diag_load=DIAG_LOAD):
    """
    MVDR: w = inv(Rn) v / (v^H inv(Rn) v)
    """
    C = Rn.shape[0]
    tr = np.trace(Rn).real
    Rn = Rn + diag_load * (tr / C + 1e-12) * np.eye(C, dtype=np.complex128)

    invRn = np.linalg.inv(Rn)
    num = invRn @ v
    denom = (v.conj().T @ invRn @ v) + 1e-12
    return (num / denom).astype(np.complex128)


def apply_mvdr_from_target_mask(X: np.ndarray, Mt: np.ndarray):
    """
    X:  (C,F,T) complex
    Mt: (F,T) float target mask

    Noise mask is Mn = 1 - Mt (more stable than "other source")
    Returns: Y (F,T) complex
    """
    C, F, T = X.shape
    Y = np.zeros((F, T), dtype=np.complex64)

    Mn = 1.0 - Mt

    for f in range(F):
        X_ft = X[:, f, :]     # (C,T)

        mt = Mt[f, :]
        mn = Mn[f, :]

        Rs = cov_from_mask_vec(X_ft, mt)
        Rn = cov_from_mask_vec(X_ft, mn)

        v = principal_eigvec(Rs)
        w = mvdr_weights(Rn, v)

        Y[f, :] = (w.conj().T @ X_ft).astype(np.complex64)

    return Y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    split = json.load(open(SPLIT_PATH, "r", encoding="utf-8"))
    test_ids = split["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    in_ch = ckpt["in_ch"]
    base = ckpt["base"]

    model = TFMaskUNet(in_ch=in_ch, base=base, n_src=2).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    for k, sid in enumerate(test_ids):
        feats = np.load(os.path.join(FEATS_DIR, f"{sid}.npy")).astype(np.float32)  # (F,T,5)
        X = np.load(os.path.join(X_DIR, f"{sid}.npy"))  # (2,F,T) complex

        x_t = torch.from_numpy(feats).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,5,F,T)
        with torch.no_grad():
            masks = model(x_t)[0].cpu().numpy().astype(np.float32)  # (2,F,T)

        # softmax masks should sum to 1, but we can re-normalize safely
        M1 = masks[0]
        M2 = masks[1]

        # optional mild sharpening (keep sum-to-1)
        if MASK_SHARPEN_P != 1.0:
            M1 = np.power(np.clip(M1, 0, 1), MASK_SHARPEN_P)
            M2 = np.power(np.clip(M2, 0, 1), MASK_SHARPEN_P)
            Z = M1 + M2 + 1e-8
            M1 /= Z
            M2 /= Z

        # Beamform each target using Mn = 1 - Mt
        Y1 = apply_mvdr_from_target_mask(X, M1)
        Y2 = apply_mvdr_from_target_mask(X, M2)

        meta = json.load(open(os.path.join(META_DIR, f"{sid}.json"), "r", encoding="utf-8"))
        length = int(meta["mix_len_samples"])

        s1_t = istft_1d(Y1, length=length)
        s2_t = istft_1d(Y2, length=length)

        mx = max(np.max(np.abs(s1_t)) + 1e-12, np.max(np.abs(s2_t)) + 1e-12)
        if mx > 0.99:
            s1_t = 0.99 * s1_t / mx
            s2_t = 0.99 * s2_t / mx

        sf.write(os.path.join(OUT_DIR, f"s1_hat_{sid}.wav"), s1_t, SR, subtype="FLOAT")
        sf.write(os.path.join(OUT_DIR, f"s2_hat_{sid}.wav"), s2_t, SR, subtype="FLOAT")

        if (k + 1) % 25 == 0:
            print(f"Beamformed {k+1}/{len(test_ids)}")

    print("Done. Wrote:", OUT_DIR)


if __name__ == "__main__":
    main()
