# torch_train.py
import os
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from torch_model import TFMaskUNet

STFT_DIR = "stft_data"
SPLIT_PATH = "splits/split.json"
OUT_DIR = "checkpoints_torch"

# ---- TUNABLES ----
BASE = 64
BATCH_SIZE = 4
LR = 1e-3
EPOCHS = 120
PATIENCE = 12

# Anti-collapse weights (IMPORTANT)
W_VAR = 0.05     # encourage non-flat masks
W_ENT = 0.01     # discourage overly peaky masks
EPS = 1e-8


class IRMDataset(Dataset):
    """
    Returns:
      feats: (5,F,T)
      irm1:  (F,T)
      irm2:  (F,T)
    """
    def __init__(self, ids, stft_dir=STFT_DIR):
        self.ids = ids
        self.feats_dir = os.path.join(stft_dir, "feats")
        self.mags_dir = os.path.join(stft_dir, "mags")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        feats = np.load(os.path.join(self.feats_dir, f"{sid}.npy")).astype(np.float32)  # (F,T,5)

        mags_path = os.path.join(self.mags_dir, f"{sid}.npz")
        if not os.path.exists(mags_path):
            raise FileNotFoundError(f"Missing {mags_path}. Run: python feature_extraction.py")

        mz = np.load(mags_path)
        irm1 = mz["irm1"].astype(np.float32)  # (F,T)
        irm2 = mz["irm2"].astype(np.float32)

        feats = torch.from_numpy(feats).permute(2, 0, 1)  # (5,F,T)
        irm1 = torch.from_numpy(irm1)                     # (F,T)
        irm2 = torch.from_numpy(irm2)

        return feats, irm1, irm2


def mask_pit_loss(pred_masks, irm1, irm2, eps=EPS):
    """
    pred_masks: (B,2,F,T) softmax outputs
    irm1/irm2:  (B,F,T) oracle targets

    Loss = PIT MSE + anti-collapse
    """
    m1 = pred_masks[:, 0]
    m2 = pred_masks[:, 1]

    # PIT (per-example)
    a = torch.mean((m1 - irm1) ** 2 + (m2 - irm2) ** 2, dim=(1, 2))
    b = torch.mean((m1 - irm2) ** 2 + (m2 - irm1) ** 2, dim=(1, 2))
    pit = torch.minimum(a, b).mean()

    # Anti-collapse variance reward (subtract)
    var = (torch.var(m1, dim=(1, 2)) + torch.var(m2, dim=(1, 2))).mean()
    # Mild entropy penalty (avoid super peaky masks that can hurt SAR)
    ent = -(m1 * torch.log(m1 + eps) + m2 * torch.log(m2 + eps)).mean()

    loss = pit - W_VAR * var + W_ENT * ent
    return loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for feats, irm1, irm2 in loader:
        feats = feats.to(device, non_blocking=True)
        irm1 = irm1.to(device, non_blocking=True)
        irm2 = irm2.to(device, non_blocking=True)

        pred = model(feats)
        loss = mask_pit_loss(pred, irm1, irm2)

        total += float(loss.item()) * feats.size(0)
        n += feats.size(0)

    return total / max(n, 1)


def gpu_mem(prefix=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / (1024 ** 2)
        r = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"{prefix}GPU mem: allocated={a:.1f}MB reserved={r:.1f}MB")


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
        torch.cuda.empty_cache()
        gpu_mem("[startup] ")

    print("Loading datasets...")
    train_ds = IRMDataset(train_ids)
    val_ds = IRMDataset(val_ids)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Alive check sample
    x0, irm10, irm20 = train_ds[0]
    in_ch = int(x0.shape[0])
    print("Sample feats:", tuple(x0.shape), "IRM:", tuple(irm10.shape))
    print(f"IRM stats: mean={float(irm10.mean()):.3f} std={float(irm10.std()):.3f}")

    print(f"Building model base={BASE} ...")
    model = TFMaskUNet(in_ch=in_ch, base=BASE, n_src=2).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # DataLoaders
    num_workers = 0  # safest on Windows
    print(f"Creating dataloaders batch={BATCH_SIZE}, num_workers={num_workers} ...")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Alive check first batch + forward
    print("Alive check: loading first batch ...")
    feats, irm1, irm2 = next(iter(train_loader))
    print("Alive check: first batch loaded feats:", tuple(feats.shape), "irm:", tuple(irm1.shape))

    feats = feats.to(device)
    irm1 = irm1.to(device)
    irm2 = irm2.to(device)

    if device.type == "cuda":
        gpu_mem("[before forward] ")

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred = model(feats)
            l0 = mask_pit_loss(pred, irm1, irm2)
    print("Alive check: forward OK, loss =", float(l0.item()))

    if device.type == "cuda":
        gpu_mem("[after forward] ")

    best_val = float("inf")
    best_path = os.path.join(OUT_DIR, "best.pt")
    final_path = os.path.join(OUT_DIR, "final.pt")
    bad = 0

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        total = 0.0
        n = 0

        for feats, irm1, irm2 in train_loader:
            feats = feats.to(device, non_blocking=True)
            irm1 = irm1.to(device, non_blocking=True)
            irm2 = irm2.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(feats)
                loss = mask_pit_loss(pred, irm1, irm2)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += float(loss.item()) * feats.size(0)
            n += feats.size(0)

        train_loss = total / max(n, 1)
        val_loss = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(f"Epoch {epoch:03d}/{EPOCHS}  {dt:.1f}s  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad = 0
            torch.save({"model_state": model.state_dict(), "in_ch": in_ch, "base": BASE}, best_path)
            print("  ✅ saved best:", best_path)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping.")
                break

    torch.save({"model_state": model.state_dict(), "in_ch": in_ch, "base": BASE}, final_path)
    print("Saved final:", final_path)


if __name__ == "__main__":
    main()
