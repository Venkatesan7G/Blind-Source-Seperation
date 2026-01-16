import os
import json
import numpy as np
import soundfile as sf
import torch

from stft_util import istft_1d

STFT_DIR = "stft_data"
SPLIT_PATH = "splits/split.json"
CKPT_PATH = "checkpoints_rd/best.pt"
OUT_DIR = "separated_rd"

X_DIR = os.path.join(STFT_DIR, "X")
META_DIR = os.path.join(STFT_DIR, "meta")

SR = 16000
EPS = 1e-8

class UnfoldedRotationDemixer(torch.nn.Module):
    def __init__(self, F: int, K: int = 8, theta_lr: float = 0.1):
        super().__init__()
        self.F = F
        self.K = K
        self.alpha = torch.nn.Parameter(torch.ones(K) * theta_lr)
        self.beta = torch.nn.Parameter(torch.ones(F))

    def forward(self, X: torch.Tensor):
        B, C, F, T = X.shape
        x1 = X[:, 0]
        x2 = X[:, 1]
        theta = torch.zeros((B, F), device=X.device, dtype=torch.float32)

        for k in range(self.K):
            ct = torch.cos(theta)[:, :, None]
            st = torch.sin(theta)[:, :, None]
            y1 = ct * x1 + st * x2
            y2 = -st * x1 + ct * x2
            cross = torch.sum(y1 * torch.conj(y2), dim=-1)
            pwr = torch.sum((y1.real**2 + y1.imag**2) + (y2.real**2 + y2.imag**2), dim=-1) + EPS
            g = (cross.imag / pwr).float()
            g = g * self.beta[None, :]
            theta = theta - self.alpha[k] * g

        ct = torch.cos(theta)[:, :, None]
        st = torch.sin(theta)[:, :, None]
        y1 = ct * x1 + st * x2
        y2 = -st * x1 + ct * x2
        return torch.stack([y1, y2], dim=1)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"Missing {SPLIT_PATH}. Run split_data.py first.")
    split = json.load(open(SPLIT_PATH, "r", encoding="utf-8"))
    test_ids = split["test"]
    print("Total test files:", len(test_ids))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Missing checkpoint {CKPT_PATH}. Train first.")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    F = ckpt["F"]
    K = ckpt["K"]
    theta_lr = ckpt["theta_lr"]

    model = UnfoldedRotationDemixer(F=F, K=K, theta_lr=theta_lr).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    for k, sid in enumerate(test_ids, start=1):
        X = np.load(os.path.join(X_DIR, f"{sid}.npy")).astype(np.complex64)  # (2,F,T)
        meta = json.load(open(os.path.join(META_DIR, f"{sid}.json"), "r", encoding="utf-8"))
        length = int(meta["mix_len_samples"])

        Xt = torch.from_numpy(X)[None, ...].to(device)  # (1,2,F,T) complex
        with torch.no_grad():
            Y = model(Xt)[0].cpu().numpy()  # (2,F,T) complex

        y1 = Y[0].astype(np.complex64)
        y2 = Y[1].astype(np.complex64)

        s1_t = istft_1d(y1, length=length)
        s2_t = istft_1d(y2, length=length)

        mx = max(np.max(np.abs(s1_t)) + 1e-12, np.max(np.abs(s2_t)) + 1e-12)
        if mx > 0.99:
            s1_t = 0.99 * s1_t / mx
            s2_t = 0.99 * s2_t / mx

        sf.write(os.path.join(OUT_DIR, f"s1_hat_{sid}.wav"), s1_t, SR, subtype="FLOAT")
        sf.write(os.path.join(OUT_DIR, f"s2_hat_{sid}.wav"), s2_t, SR, subtype="FLOAT")

        if k % 25 == 0 or k == len(test_ids):
            print(f"Separated {k}/{len(test_ids)}")

    print("Done. Wrote:", OUT_DIR)

if __name__ == "__main__":
    main()
