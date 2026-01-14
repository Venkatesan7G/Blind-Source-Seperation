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


class BSSNpyDataset(Dataset):
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
            raise FileNotFoundError(
                f"Missing {mags_path}. Re-run: python feature_extraction.py"
            )
        mz = np.load(mags_path)
        mix_mag = mz["mix_mag"].astype(np.float32)  # (F,T) |X_mic1|
        s1_mag = mz["s1_mag"].astype(np.float32)    # (F,T) |S1_mic1|
        s2_mag = mz["s2_mag"].astype(np.float32)    # (F,T) |S2_mic1|

        # to torch
        feats = torch.from_numpy(feats).permute(2, 0, 1)  # (5,F,T)
        mix_mag = torch.from_numpy(mix_mag)               # (F,T)
        s1_mag = torch.from_numpy(s1_mag)                 # (F,T)
        s2_mag = torch.from_numpy(s2_mag)                 # (F,T)

        return feats, mix_mag, s1_mag, s2_mag


def mag_recon_loss(pred_masks, mix_mag, s1_mag, s2_mag):
    """
    pred_masks: (B,2,F,T) softmax masks
    mix_mag:    (B,F,T)
    s1_mag/s2_mag: (B,F,T)
    """
    m1 = pred_masks[:, 0, :, :]
    m2 = pred_masks[:, 1, :, :]

    s1_hat = m1 * mix_mag
    s2_hat = m2 * mix_mag

    loss1 = torch.mean((s1_hat - s1_mag) ** 2)
    loss2 = torch.mean((s2_hat - s2_mag) ** 2)
    return loss1 + loss2


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for feats, mix_mag, s1_mag, s2_mag in loader:
            feats = feats.to(device, non_blocking=True)
            mix_mag = mix_mag.to(device, non_blocking=True)
            s1_mag = s1_mag.to(device, non_blocking=True)
            s2_mag = s2_mag.to(device, non_blocking=True)

            pred = model(feats)
            loss = mag_recon_loss(pred, mix_mag, s1_mag, s2_mag)

            total += float(loss.item()) * feats.size(0)
            n += feats.size(0)
    return total / max(n, 1)


def gpu_mem(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"{prefix}GPU mem: allocated={allocated:.1f}MB reserved={reserved:.1f}MB")


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
    train_ds = BSSNpyDataset(train_ids)
    val_ds = BSSNpyDataset(val_ids)
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    # Infer shapes from one sample
    x0, mix0, s10, s20 = train_ds[0]
    in_ch = x0.shape[0]
    print("Sample shapes feats:", tuple(x0.shape), "mix_mag:", tuple(mix0.shape))

    # ---- MODEL SIZE ----
    base = 48
    print(f"Building model (base={base}) ...")
    model = TFMaskUNet(in_ch=in_ch, base=base, n_src=2).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # AMP (safe)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ---- WINDOWS-SAFE LOADER (prevents hangs) ----
    batch_size = 4  # start small; you can increase to 8 later if stable
    print(f"Creating dataloaders (batch_size={batch_size}, num_workers=0) ...")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # IMPORTANT for Windows stability
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Alive check: load first batch
    print("Loading first batch (alive check)...")
    feats, mix_mag, s1_mag, s2_mag = next(iter(train_loader))
    print("First batch loaded.")
    print("  feats:", tuple(feats.shape), "mix_mag:", tuple(mix_mag.shape))

    feats = feats.to(device)
    mix_mag = mix_mag.to(device)
    s1_mag = s1_mag.to(device)
    s2_mag = s2_mag.to(device)

    if device.type == "cuda":
        gpu_mem("[before first forward] ")

    # Alive check: one forward pass
    print("Running first forward pass (alive check)...")
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred = model(feats)
            loss0 = mag_recon_loss(pred, mix_mag, s1_mag, s2_mag)
    print("First forward OK. Loss:", float(loss0.item()))
    if device.type == "cuda":
        gpu_mem("[after first forward] ")

    # Training loop
    best_val = float("inf")
    best_path = os.path.join(OUT_DIR, "best.pt")
    final_path = os.path.join(OUT_DIR, "final.pt")

    epochs = 60
    patience = 10
    bad = 0

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total = 0.0
        n = 0

        for feats, mix_mag, s1_mag, s2_mag in train_loader:
            feats = feats.to(device, non_blocking=True)
            mix_mag = mix_mag.to(device, non_blocking=True)
            s1_mag = s1_mag.to(device, non_blocking=True)
            s2_mag = s2_mag.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(feats)
                loss = mag_recon_loss(pred, mix_mag, s1_mag, s2_mag)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += float(loss.item()) * feats.size(0)
            n += feats.size(0)

        train_loss = total / max(n, 1)
        val_loss = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(f"Epoch {epoch:02d}/{epochs} - {dt:.1f}s - train_loss {train_loss:.4f} - val_loss {val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad = 0
            torch.save(
                {"model_state": model.state_dict(), "in_ch": in_ch, "base": base},
                best_path,
            )
            print("  ✅ saved best:", best_path)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    torch.save(
        {"model_state": model.state_dict(), "in_ch": in_ch, "base": base},
        final_path,
    )
    print("Saved final:", final_path)


if __name__ == "__main__":
    main()
