# torch_train.py
import os
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from torch_model import UnfoldedComplexGivensDemixer

STFT_DIR = "stft_data"
SPLIT_PATH = "splits/split.json"
OUT_DIR = "checkpoints_rd"

# ---- TUNABLES ----
EPOCHS = 120
BATCH_SIZE = 6
LR = 3e-4
PATIENCE = 12

K_ITERS = 8
THETA_LR = 0.15
PHI_LR = 0.05

# loss weights
LAMBDA_UNITARY = 1.0  # keep stable (small)
GRAD_CLIP = 5.0

EPS = 1e-8


def cplx_mse_per_ex(a: torch.Tensor, b: torch.Tensor):
    """
    a,b: (B,F,T) complex
    returns: (B,) per-example complex MSE averaged over (F,T)
    """
    d_re = (a.real - b.real) ** 2
    d_im = (a.imag - b.imag) ** 2
    return torch.mean(d_re + d_im, dim=(1, 2))


def pit_complex_loss(Y: torch.Tensor, S1: torch.Tensor, S2: torch.Tensor):
    """
    PER-EXAMPLE PIT (THIS FIXES YOUR CORE ISSUE)
    Y:  (B,2,F,T) complex
    S1,S2: (B,F,T) complex
    returns scalar
    """
    y1 = Y[:, 0]
    y2 = Y[:, 1]

    La = cplx_mse_per_ex(y1, S1) + cplx_mse_per_ex(y2, S2)  # (B,)
    Lb = cplx_mse_per_ex(y1, S2) + cplx_mse_per_ex(y2, S1)  # (B,)

    L = torch.minimum(La, Lb)  # (B,)
    return torch.mean(L)


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
        S1 = S["S1"].astype(np.complex64)  # (F,T)
        S2 = S["S2"].astype(np.complex64)

        X = torch.from_numpy(X)   # complex
        S1 = torch.from_numpy(S1)
        S2 = torch.from_numpy(S2)
        return X, S1, S2


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
            loss = pit_complex_loss(Y, S1, S2) + LAMBDA_UNITARY * model.unitary_penalty()

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

    model = UnfoldedComplexGivensDemixer(
        F=F, K=K_ITERS, theta_lr=THETA_LR, phi_lr=PHI_LR, smooth_ks=9
    ).to(device)

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

            loss = pit_complex_loss(Y, S1, S2) + LAMBDA_UNITARY * model.unitary_penalty()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            total += float(loss.item()) * X.size(0)
            n += X.size(0)

        train_loss = total / max(n, 1)
        val_loss = evaluate(model, val_loader, device)
        dt = time.time() - t0

        at = model.alpha_theta.detach().cpu().numpy().round(4)
        ap = model.alpha_phi.detach().cpu().numpy().round(4)

        print(f"Epoch {epoch:03d}/{EPOCHS}  {dt:.1f}s  train={train_loss:.6f}  val={val_loss:.6f}")
        print(f"  alpha_theta={at}")
        print(f"  alpha_phi  ={ap}")

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            bad = 0
            torch.save(
                {"model_state": model.state_dict(), "F": F, "K": K_ITERS,
                 "theta_lr": THETA_LR, "phi_lr": PHI_LR},
                best_path
            )
            print("  ✅ saved best:", best_path)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping.")
                break

    torch.save(
        {"model_state": model.state_dict(), "F": F, "K": K_ITERS,
         "theta_lr": THETA_LR, "phi_lr": PHI_LR},
        final_path
    )
    print("Saved final:", final_path)


if __name__ == "__main__":
    main()
