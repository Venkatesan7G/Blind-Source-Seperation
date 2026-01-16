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
OUT_DIR = "separated"

META_DIR = os.path.join(STFT_DIR, "meta")
FEATS_DIR = os.path.join(STFT_DIR, "feats")
X_DIR = os.path.join(STFT_DIR, "X")

SR = 16000
EPS = 1e-8

# ---- NEW: stabilization knobs ----
MASK_FLOOR = 0.02     # try 0.01..0.05
MASK_CEIL = 0.98
USE_MIXTURE_CONSISTENCY = True  # Option A


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"Missing {SPLIT_PATH}. Run split_data.py first.")

    split = json.load(open(SPLIT_PATH, "r", encoding="utf-8"))
    test_ids = split["test"]
    print(f"Total test files: {len(test_ids)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Missing checkpoint {CKPT_PATH}. Train first.")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    in_ch = ckpt["in_ch"]
    base = ckpt["base"]

    model = TFMaskUNet(in_ch=in_ch, base=base, n_src=2).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    for k, sid in enumerate(test_ids):
        feats_path = os.path.join(FEATS_DIR, f"{sid}.npy")
        X_path = os.path.join(X_DIR, f"{sid}.npy")
        meta_path = os.path.join(META_DIR, f"{sid}.json")

        feats = np.load(feats_path).astype(np.float32)  # (F,T,5)
        X = np.load(X_path)                              # (2,F,T) complex
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        length = int(meta["mix_len_samples"])

        x = torch.from_numpy(feats).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,5,F,T)

        with torch.no_grad():
            masks = model(x)[0].cpu().numpy().astype(np.float32)  # (2,F,T)

        # ---- Option B: stabilize masks (clip + renormalize) ----
        M1 = masks[0]
        M2 = masks[1]

        M1 = np.clip(M1, MASK_FLOOR, MASK_CEIL)
        M2 = np.clip(M2, MASK_FLOOR, MASK_CEIL)
        Z = M1 + M2 + 1e-8
        M1 /= Z
        M2 /= Z

        # ---- Reconstruct using mic1 mixture STFT ----
        X1 = X[0].astype(np.complex64)  # (F,T)

        S1_hat = (M1 * X1).astype(np.complex64)
        S2_hat = (M2 * X1).astype(np.complex64)

        # ---- Balanced mixture consistency ----
        R = (X1 - (S1_hat + S2_hat)).astype(np.complex64)
        S1_hat = S1_hat + 0.5 * R
        S2_hat = S2_hat + 0.5 * R


        s1_t = istft_1d(S1_hat, length=length)
        s2_t = istft_1d(S2_hat, length=length)

        # normalize to avoid clipping
        mx = max(np.max(np.abs(s1_t)) + 1e-12, np.max(np.abs(s2_t)) + 1e-12)
        if mx > 0.99:
            s1_t = 0.99 * s1_t / mx
            s2_t = 0.99 * s2_t / mx

        sf.write(os.path.join(OUT_DIR, f"s1_hat_{sid}.wav"), s1_t, SR, subtype="FLOAT")
        sf.write(os.path.join(OUT_DIR, f"s2_hat_{sid}.wav"), s2_t, SR, subtype="FLOAT")

        if (k + 1) % 25 == 0 or (k + 1) == len(test_ids):
            print(
                f"Separated {k+1}/{len(test_ids)} | "
                f"M1 mean={float(M1.mean()):.3f} std={float(M1.std()):.3f}"
            )

    print("Done. Wrote:", OUT_DIR)


if __name__ == "__main__":
    main()
