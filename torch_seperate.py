import os
import json
import numpy as np
import soundfile as sf
import torch

from torch_model import UnfoldedComplexGivensDemixer
from stft_util import istft_1d

STFT_DIR = "stft_data"
SPLIT_PATH = "splits/split.json"

# Use best.pt for evaluation unless you explicitly want final.pt
CKPT_PATH = "checkpoints_rd/best.pt"

OUT_DIR = "separated_rd"
META_DIR = os.path.join(STFT_DIR, "meta")
X_DIR = os.path.join(STFT_DIR, "X")

SR = 16000


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"Missing {SPLIT_PATH}. Run split_data.py first.")

    split = json.load(open(SPLIT_PATH, "r", encoding="utf-8"))
    test_ids = split["test"]
    print(f"Total test files: {len(test_ids)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Missing checkpoint {CKPT_PATH}. Train first.")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    F = int(ckpt["F"])
    K = int(ckpt["K"])
    theta_lr = float(ckpt.get("theta_lr", 0.15))
    phi_lr = float(ckpt.get("phi_lr", 0.05))

    model = UnfoldedComplexGivensDemixer(
        F=F, K=K, theta_lr=theta_lr, phi_lr=phi_lr, smooth_ks=9
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    for k, sid in enumerate(test_ids):
        x_path = os.path.join(X_DIR, f"{sid}.npy")
        meta_path = os.path.join(META_DIR, f"{sid}.json")

        if not os.path.exists(x_path):
            print(f"[WARN] missing X for id={sid}: {x_path}")
            continue
        if not os.path.exists(meta_path):
            print(f"[WARN] missing meta for id={sid}: {meta_path}")
            continue

        X = np.load(x_path)  # (2,F,T) complex64
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        length = int(meta["mix_len_samples"])

        # Torch complex input: (1,2,F,T)
        Xt = torch.from_numpy(X).to(device)
        if Xt.ndim != 3 or Xt.shape[0] != 2:
            raise RuntimeError(f"Bad X shape for {sid}: {tuple(Xt.shape)}")
        Xt = Xt.unsqueeze(0)  # (1,2,F,T)

        with torch.no_grad():
            Y = model(Xt)[0]  # (2,F,T) complex

        Y1 = Y[0].detach().cpu().numpy().astype(np.complex64)
        Y2 = Y[1].detach().cpu().numpy().astype(np.complex64)

        s1_t = istft_1d(Y1, length=length)
        s2_t = istft_1d(Y2, length=length)

        # normalize to avoid clipping
        mx = max(np.max(np.abs(s1_t)) + 1e-12, np.max(np.abs(s2_t)) + 1e-12)
        if mx > 0.99:
            s1_t = 0.99 * s1_t / mx
            s2_t = 0.99 * s2_t / mx

        sf.write(os.path.join(OUT_DIR, f"s1_hat_{sid}.wav"), s1_t, SR, subtype="FLOAT")
        sf.write(os.path.join(OUT_DIR, f"s2_hat_{sid}.wav"), s2_t, SR, subtype="FLOAT")

        if (k + 1) % 25 == 0 or (k + 1) == len(test_ids):
            # these alphas exist in your training script and are printed there too :contentReference[oaicite:3]{index=3}
            at = model.alpha_theta.detach().cpu().numpy()
            ap = model.alpha_phi.detach().cpu().numpy()
            print(
                f"Separated {k+1}/{len(test_ids)} | "
                f"alpha_theta mean={float(at.mean()):.4f} | alpha_phi mean={float(ap.mean()):.4f}"
            )

    print("Done. Wrote:", OUT_DIR)


if __name__ == "__main__":
    main()
