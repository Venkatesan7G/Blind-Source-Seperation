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
EPOCHS = 80
PATIENCE = 12

ALPHA_CONS = 0.2
EPS = 1e-8


class BSSMagDataset(Dataset):
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
        mix_mag = mz["mix_mag"].astype(np.float32)  # (F,T)
        s1_mag = mz["s1_mag"].astype(np.float32)
        s2_mag = mz["s2_mag"].astype(np.float32)

        feats = torch.from_numpy(feats).permute(2, 0, 1)  # (5,F,T)
        mix_mag = torch.from_numpy(mix_mag)              # (F,T)
        s1_mag = torch.from_numpy(s1_mag)
        s2_mag = torch.from_numpy(s2_mag)

        return feats, mix_mag, s1_mag, s2_mag


def _per_example_logmse(a, b, eps=EPS):
    """
    a,b: (B,F,T)
    returns: (B,)  mean over (F,T) per example
    """
    da = torch.log1p(a + eps) - torch.log1p(b + eps)
    return torch.mean(da * da, dim=(1, 2))


def _per_example_mse(a, b):
    """
    a,b: (B,F,T)
    returns: (B,)
    """
    d = a - b
    return torch.mean(d * d, dim=(1, 2))


def mag_loss(pred_masks, mix_mag, s1_mag, s2_mag, alpha_cons=ALPHA_CONS, eps=EPS):
    """
    Correct PIT loss: choose best permutation PER EXAMPLE (not one scalar for whole batch).
    pred_masks: (B,2,F,T) softmax
    mix_mag,s1_mag,s2_mag: (B,F,T)
    """
    m1 = pred_masks[:, 0]  # (B,F,T)
    m2 = pred_masks[:, 1]

    # assignment A
    s1_hat_a = m1 * mix_mag
    s2_hat_a = m2 * mix_mag
    loss_a = _per_example_logmse(s1_hat_a, s1_mag, eps) + _per_example_logmse(s2_hat_a, s2_mag, eps)

    # assignment B (swap)
    s1_hat_b = m2 * mix_mag
    s2_hat_b = m1 * mix_mag
    loss_b = _per_example_logmse(s1_hat_b, s1_mag, eps) + _per_example_logmse(s2_hat_b, s2_mag, eps)

    # pick best per example
    use_b = (loss_b < loss_a).float()  # (B,)
    loss = (1.0 - use_b) * loss_a + use_b * loss_b  # (B,)

    # mixture consistency per example, matched to chosen permutation
    cons_a = _per_example_mse(s1_hat_a + s2_hat_a, mix_mag)
    cons_b = _per_example_mse(s1_hat_b + s2_hat_b, mix_mag)
    cons = (1.0 - use_b) * cons_a + use_b * cons_b  # (B,)

    return torch.mean(loss + alpha_cons * cons)


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
            loss = mag_loss(pred, mix_mag, s1_mag, s2_mag)

            total += float(loss.item()) * feats.size(0)
            n += feats.size(0)
    return total / max(n, 1)


def gpu_mem(prefix=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / (1024**2)
        r = torch.cuda.memory_reserved() / (1024**2)
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
    train_ds = BSSMagDataset(train_ids)
    val_ds = BSSMagDataset(val_ids)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Alive check sample
    x0, mix0, s10, s20 = train_ds[0]
    in_ch = x0.shape[0]
    print("Sample feats:", tuple(x0.shape), "mix_mag:", tuple(mix0.shape))

    print(f"Building model base={BASE} ...")
    model = TFMaskUNet(in_ch=in_ch, base=BASE, n_src=2).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print(f"Creating dataloaders batch={BATCH_SIZE}, num_workers=0 ...")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                            pin_memory=(device.type == "cuda"))

    # Alive check first batch + forward
    print("Alive check: loading first batch ...")
    feats, mix_mag, s1_mag, s2_mag = next(iter(train_loader))
    print("Alive check: first batch loaded:", tuple(feats.shape))
    feats = feats.to(device)
    mix_mag = mix_mag.to(device)
    s1_mag = s1_mag.to(device)
    s2_mag = s2_mag.to(device)
    if device.type == "cuda":
        gpu_mem("[before forward] ")
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred = model(feats)
            l0 = mag_loss(pred, mix_mag, s1_mag, s2_mag)
    print("Alive check: forward OK, loss=", float(l0.item()))
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

        for feats, mix_mag, s1_mag, s2_mag in train_loader:
            feats = feats.to(device, non_blocking=True)
            mix_mag = mix_mag.to(device, non_blocking=True)
            s1_mag = s1_mag.to(device, non_blocking=True)
            s2_mag = s2_mag.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(feats)
                loss = mag_loss(pred, mix_mag, s1_mag, s2_mag)

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
