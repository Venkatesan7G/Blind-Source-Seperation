import os
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

STFT_DIR = "stft_data"
SPLIT_PATH = "splits/split.json"
OUT_DIR = "checkpoints_rd"

# ---- TUNABLES ----
EPOCHS = 120
BATCH_SIZE = 6
LR = 3e-4
PATIENCE = 12

K_ITERS = 8          # unfolded steps (layers)
THETA_LR = 0.15      # initial step size, learnable per-iter in the model
EPS = 1e-8

def cplx_mse(a: torch.Tensor, b: torch.Tensor):
    # a,b complex: (...,)
    return torch.mean((a.real - b.real) ** 2 + (a.imag - b.imag) ** 2)

class STFTDemixDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids
        self.X_dir = os.path.join(STFT_DIR, "X")
        self.S_dir = os.path.join(STFT_DIR, "S")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        X = np.load(os.path.join(self.X_dir, f"{sid}.npy"))  # (2,F,T) complex64
        S = np.load(os.path.join(self.S_dir, f"{sid}.npz"))
        S1 = S["S1"]  # (F,T) complex64
        S2 = S["S2"]
        # torch complex
        X = torch.from_numpy(X)    # complex64
        S1 = torch.from_numpy(S1)
        S2 = torch.from_numpy(S2)
        return X, S1, S2

class UnfoldedRotationDemixer(torch.nn.Module):
    """
    2ch demixer via per-frequency rotation angle theta(f):
      y1 =  cosθ * x1 + sinθ * x2
      y2 = -sinθ * x1 + cosθ * x2

    Update theta iteratively using decorrelation signal.
    This is deep-unfolding: K iterations with learnable step sizes.
    """
    def __init__(self, F: int, K: int = 8, theta_lr: float = 0.1):
        super().__init__()
        self.F = F
        self.K = K
        # learnable step per iteration
        self.alpha = torch.nn.Parameter(torch.ones(K) * theta_lr)

        # optional per-frequency gain on the update (learnable)
        self.beta = torch.nn.Parameter(torch.ones(F))

    def forward(self, X: torch.Tensor):
        """
        X: (B,2,F,T) complex
        returns Y: (B,2,F,T) complex
        """
        B, C, F, T = X.shape
        assert C == 2 and F == self.F

        x1 = X[:, 0]  # (B,F,T)
        x2 = X[:, 1]

        theta = torch.zeros((B, F), device=X.device, dtype=torch.float32)

        for k in range(self.K):
            ct = torch.cos(theta)[:, :, None]  # (B,F,1)
            st = torch.sin(theta)[:, :, None]

            y1 = ct * x1 + st * x2
            y2 = -st * x1 + ct * x2

            # decorrelation signal g(f): imag(cross-power) / power
            # cross = sum_t y1 * conj(y2)
            cross = torch.sum(y1 * torch.conj(y2), dim=-1)  # (B,F) complex
            pwr = torch.sum((y1.real**2 + y1.imag**2) + (y2.real**2 + y2.imag**2), dim=-1) + EPS  # (B,F)

            g = (cross.imag / pwr).float()  # (B,F)

            # learned per-freq scale
            g = g * self.beta[None, :]

            theta = theta - self.alpha[k] * g

        # final apply
        ct = torch.cos(theta)[:, :, None]
        st = torch.sin(theta)[:, :, None]
        y1 = ct * x1 + st * x2
        y2 = -st * x1 + ct * x2
        Y = torch.stack([y1, y2], dim=1)  # (B,2,F,T)
        return Y

def pit_loss(Y: torch.Tensor, S1: torch.Tensor, S2: torch.Tensor):
    """
    Y:  (B,2,F,T) complex
    S1,S2: (B,F,T) complex
    """
    y1 = Y[:, 0]
    y2 = Y[:, 1]

    # assignment A: y1->S1, y2->S2
    La = cplx_mse(y1, S1) + cplx_mse(y2, S2)
    # assignment B: y1->S2, y2->S1
    Lb = cplx_mse(y1, S2) + cplx_mse(y2, S1)
    return torch.minimum(La, Lb)

def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for X, S1, S2 in loader:
            X = X.to(device)
            S1 = S1.to(device)
            S2 = S2.to(device)

            Y = model(X)
            loss = pit_loss(Y, S1, S2)
            total += float(loss.item()) * X.size(0)
            n += X.size(0)
    return total / max(n, 1)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"Missing {SPLIT_PATH}. Run: python split_data.py")
    split = json.load(open(SPLIT_PATH, "r", encoding="utf-8"))
    train_ids = split["train"]
    val_ids = split["val"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    train_ds = STFTDemixDataset(train_ids)
    val_ds = STFTDemixDataset(val_ids)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # infer F from one sample
    X0, S10, S20 = train_ds[0]
    F = X0.shape[1]
    print("Sample X shape:", tuple(X0.shape), "F=", F)

    model = UnfoldedRotationDemixer(F=F, K=K_ITERS, theta_lr=THETA_LR).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_val = float("inf")
    bad = 0
    best_path = os.path.join(OUT_DIR, "best.pt")
    final_path = os.path.join(OUT_DIR, "final.pt")

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        total = 0.0
        n = 0

        for X, S1, S2 in train_loader:
            X = X.to(device)
            S1 = S1.to(device)
            S2 = S2.to(device)

            opt.zero_grad(set_to_none=True)
            Y = model(X)
            loss = pit_loss(Y, S1, S2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total += float(loss.item()) * X.size(0)
            n += X.size(0)

        train_loss = total / max(n, 1)
        val_loss = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(f"Epoch {epoch:03d}/{EPOCHS}  {dt:.1f}s  train={train_loss:.6f}  val={val_loss:.6f}")
        print(f"  alpha={model.alpha.data.detach().cpu().numpy().round(4)}")

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            bad = 0
            torch.save(
                {"model_state": model.state_dict(),
                 "F": F, "K": K_ITERS, "theta_lr": THETA_LR},
                best_path
            )
            print("  ✅ saved best:", best_path)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping.")
                break

    torch.save(
        {"model_state": model.state_dict(),
         "F": F, "K": K_ITERS, "theta_lr": THETA_LR},
        final_path
    )
    print("Saved final:", final_path)

if __name__ == "__main__":
    main()
