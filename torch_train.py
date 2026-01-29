# torch_train_FIXED_v2.py - CORRECTED DISCRIMINATION WEIGHT
import os
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from torch_model_fixed import UnfoldedComplexGivensDemixerFixed

STFT_DIR = "stft_data"
SPLIT_PATH = "splits/split.json"
OUT_DIR = "checkpoints_fixed"

# ==== CORRECTED HYPERPARAMETERS ====
EPOCHS = 350
BATCH_SIZE = 8
LR = 5e-4
PATIENCE = 50

K_ITERS = 12
THETA_LR = 0.12
PHI_LR = 0.04

LAMBDA_UNITARY = 1.0
ALPHA_DISCRIM = 0.05       # ⭐ CORRECTED: Was 0.3, now 0.05 (much lower!)
GRAD_CLIP = 2.0

EPS = 1e-8


def cplx_mse_per_ex(a: torch.Tensor, b: torch.Tensor):
    d_re = (a.real - b.real) ** 2
    d_im = (a.imag - b.imag) ** 2
    return torch.mean(d_re + d_im, dim=(1, 2))


def pit_complex_loss(Y: torch.Tensor, S1: torch.Tensor, S2: torch.Tensor):
    y1 = Y[:, 0]
    y2 = Y[:, 1]
    La = cplx_mse_per_ex(y1, S1) + cplx_mse_per_ex(y2, S2)
    Lb = cplx_mse_per_ex(y1, S2) + cplx_mse_per_ex(y2, S1)
    L = torch.minimum(La, Lb)
    return torch.mean(L)


def discriminative_pit_loss(Y, S1, S2, alpha=ALPHA_DISCRIM):
    """Enhanced PIT loss with CORRECTED discrimination weight"""
    loss_recon = pit_complex_loss(Y, S1, S2)
    
    y1 = Y[:, 0]
    y2 = Y[:, 1]
    
    mag1 = torch.abs(y1)
    mag2 = torch.abs(y2)
    
    mag1_norm = mag1 / (torch.mean(mag1, dim=-1, keepdim=True) + EPS)
    mag2_norm = mag2 / (torch.mean(mag2, dim=-1, keepdim=True) + EPS)
    
    correlation = torch.mean(mag1_norm * mag2_norm, dim=-1)
    loss_discrim = torch.mean(torch.relu(correlation))
    
    total_loss = loss_recon + alpha * loss_discrim
    
    return total_loss, loss_recon.item(), loss_discrim.item()


class STFTDemixDataset(Dataset):
    def __init__(self, ids, augment=False):
        self.ids = ids
        self.X_dir = os.path.join(STFT_DIR, "X")
        self.S_dir = os.path.join(STFT_DIR, "S")
        self.augment = augment
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        sid = self.ids[idx]
        X = np.load(os.path.join(self.X_dir, f"{sid}.npy"))
        S = np.load(os.path.join(self.S_dir, f"{sid}.npz"))
        S1 = S["S1"].astype(np.complex64)
        S2 = S["S2"].astype(np.complex64)
        
        if self.augment:
            X, S1, S2 = self._augment(X, S1, S2)
        
        return torch.from_numpy(X), torch.from_numpy(S1), torch.from_numpy(S2)
    
    def _augment(self, X, S1, S2):
        # Channel swap
        if np.random.rand() < 0.5:
            X = X[[1, 0], :, :]
            S1, S2 = S2, S1
        
        # Time shift
        shift = np.random.randint(-30, 31)
        if shift != 0:
            X = np.roll(X, shift, axis=-1)
            S1 = np.roll(S1, shift, axis=-1)
            S2 = np.roll(S2, shift, axis=-1)
        
        # Phase rotation
        phase_shift = np.random.uniform(-np.pi, np.pi)
        X[1, :, :] *= np.exp(1j * phase_shift)
        
        # Magnitude scaling
        gain1 = np.random.uniform(0.8, 1.2)
        gain2 = np.random.uniform(0.8, 1.2)
        X[0, :, :] *= gain1
        X[1, :, :] *= gain2
        avg_gain = (gain1 + gain2) / 2
        S1 *= avg_gain
        S2 *= avg_gain
        
        # Frequency masking
        if np.random.rand() < 0.3:
            f_start = np.random.randint(0, max(1, X.shape[1] - 25))
            f_width = np.random.randint(5, 15)
            X[:, f_start:f_start+f_width, :] *= 0.5
        
        # Time masking
        if np.random.rand() < 0.3:
            t_start = np.random.randint(0, max(1, X.shape[2] - 50))
            t_width = np.random.randint(10, 30)
            X[:, :, t_start:t_start+t_width] *= 0.5
        
        # Additive noise
        if np.random.rand() < 0.2:
            noise_level = np.random.uniform(0.001, 0.01)
            noise = (np.random.randn(*X.shape) + 1j * np.random.randn(*X.shape)).astype(np.complex64)
            X += noise_level * noise
        
        return X.astype(np.complex64), S1.astype(np.complex64), S2.astype(np.complex64)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_discrim = 0.0
    n = 0
    
    with torch.no_grad():
        for X, S1, S2 in loader:
            X, S1, S2 = X.to(device), S1.to(device), S2.to(device)
            Y = model(X)
            
            loss, recon, discrim = discriminative_pit_loss(Y, S1, S2)
            loss += LAMBDA_UNITARY * model.unitary_penalty()
            
            total_loss += loss.item() * X.size(0)
            total_recon += recon * X.size(0)
            total_discrim += discrim * X.size(0)
            n += X.size(0)
    
    return total_loss / n, total_recon / n, total_discrim / n


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"Missing {SPLIT_PATH}. Run split_data.py first.")
    
    split = json.load(open(SPLIT_PATH, "r", encoding="utf-8"))
    train_ids = split["train"]
    val_ids = split["val"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("TRAINING CONFIGURATION - CORRECTED")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    
    train_ds = STFTDemixDataset(train_ids, augment=True)
    val_ds = STFTDemixDataset(val_ids, augment=False)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)} (with augmentation)")
    print(f"  Val:   {len(val_ds)} (no augmentation)")
    
    X0, S10, S20 = train_ds[0]
    F = X0.shape[1]
    print(f"\nSTFT shape: {tuple(X0.shape)}, F={F}")
    
    model = UnfoldedComplexGivensDemixerFixed(
        F=F, K=K_ITERS, theta_lr=THETA_LR, phi_lr=PHI_LR, smooth_ks=9
    ).to(device)
    
    print(f"\nModel parameters:")
    print(f"  K: {K_ITERS}, theta_lr: {THETA_LR}, phi_lr: {PHI_LR}")
    print(f"  ALPHA_DISCRIM: {ALPHA_DISCRIM} ⭐ CORRECTED (was 0.3, now 0.05)")
    
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    best_val = float("inf")
    bad = 0
    best_path = os.path.join(OUT_DIR, "best.pt")
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        
        total_loss = 0.0
        total_recon = 0.0
        total_discrim = 0.0
        n = 0
        
        for X, S1, S2 in train_loader:
            X, S1, S2 = X.to(device), S1.to(device), S2.to(device)
            
            opt.zero_grad(set_to_none=True)
            Y = model(X)
            
            loss, recon, discrim = discriminative_pit_loss(Y, S1, S2)
            loss += LAMBDA_UNITARY * model.unitary_penalty()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            
            total_loss += loss.item() * X.size(0)
            total_recon += recon * X.size(0)
            total_discrim += discrim * X.size(0)
            n += X.size(0)
        
        train_loss = total_loss / n
        train_recon = total_recon / n
        train_discrim = total_discrim / n
        
        val_loss, val_recon, val_discrim = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        
        dt = time.time() - t0
        
        print(f"Epoch {epoch:03d}/{EPOCHS}  {dt:.1f}s")
        print(f"  Train: loss={train_loss:.6f} recon={train_recon:.6f} discrim={train_discrim:.6f}")
        print(f"  Val:   loss={val_loss:.6f} recon={val_recon:.6f} discrim={val_discrim:.6f}")
        
        if epoch % 10 == 0:
            with torch.no_grad():
                sample_X, sample_S1, sample_S2 = next(iter(val_loader))
                sample_Y = model(sample_X[:1].to(device))
                
                y1_mag = torch.abs(sample_Y[0, 0]).cpu().numpy()
                y2_mag = torch.abs(sample_Y[0, 1]).cpu().numpy()
                
                mag_diff = np.abs(y1_mag - y2_mag).mean()
                correlation = np.corrcoef(y1_mag.flatten(), y2_mag.flatten())[0, 1]
                
                print(f"  >>> Output stats: |Y1-Y2|={mag_diff:.4f}, corr={correlation:.3f}")
                print(f"      (Want: mag_diff > 0.3, corr < 0.5)")
        
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            bad = 0
            torch.save({
                "model_state": model.state_dict(),
                "F": F, "K": K_ITERS,
                "theta_lr": THETA_LR, "phi_lr": PHI_LR,
                "epoch": epoch, "val_loss": val_loss
            }, best_path)
            print(f"  ✅ Saved best model: {best_path}")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"\nEarly stopping after {epoch} epochs")
                break
        
        print()
    
    print("=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()