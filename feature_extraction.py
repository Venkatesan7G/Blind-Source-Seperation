import os
import json
import numpy as np
import soundfile as sf

from stft_util import stft_stereo, crop_F, pad_or_trim_T, N_BINS

# Output config
OUT_DIR = "stft_data"
TARGET_F = 256          # crop 257 -> 256 to simplify pooling
TARGET_T = 620          # fixed frames
EPS = 1e-8


def ensure_dirs():
    # Added "mags" for magnitude reconstruction targets
    for sub in ["X", "S1", "S2", "feats", "masks", "mags", "meta"]:
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)


def compute_features_and_masks_and_mags(X: np.ndarray, S1: np.ndarray, S2: np.ndarray):
    """
    X, S1, S2: complex STFT (2,F,T) fullband (typically 257 bins before cropping)

    Returns:
      Xc, S1c, S2c: complex (2, TARGET_F, TARGET_T)
      feats: (TARGET_F, TARGET_T, 5)
      masks: (TARGET_F, TARGET_T, 2) ideal ratio masks at mic1
      mags: dict of float32 arrays (TARGET_F, TARGET_T):
            mix_mag, s1_mag, s2_mag (all at mic1)
    """
    # Crop F first (keep low freqs)
    Xc = crop_F(X, TARGET_F)
    S1c = crop_F(S1, TARGET_F)
    S2c = crop_F(S2, TARGET_F)

    # Pad/trim T
    Xc = pad_or_trim_T(Xc, TARGET_T)
    S1c = pad_or_trim_T(S1c, TARGET_T)
    S2c = pad_or_trim_T(S2c, TARGET_T)

    # Mixture features from both mics
    X1 = Xc[0]  # (F,T) complex mic1
    X2 = Xc[1]  # (F,T) complex mic2

    mag1 = np.abs(X1).astype(np.float32)
    mag2 = np.abs(X2).astype(np.float32)
    ipd = (np.angle(X1) - np.angle(X2)).astype(np.float32)
    ild = (np.log(mag1 + EPS) - np.log(mag2 + EPS)).astype(np.float32)

    feats = np.stack([mag1, mag2, np.cos(ipd), np.sin(ipd), ild], axis=-1).astype(np.float32)  # (F,T,5)

    # Ideal Ratio Masks (mic1)
    S1m1 = np.abs(S1c[0]).astype(np.float32)
    S2m1 = np.abs(S2c[0]).astype(np.float32)
    denom = (S1m1 + S2m1 + EPS).astype(np.float32)

    M1 = (S1m1 / denom).astype(np.float32)
    M2 = (S2m1 / denom).astype(np.float32)
    masks = np.stack([M1, M2], axis=-1).astype(np.float32)  # (F,T,2)

    mags = {
        "mix_mag": mag1.astype(np.float32),  # |X_mic1|
        "s1_mag": S1m1.astype(np.float32),   # |S1_mic1|
        "s2_mag": S2m1.astype(np.float32),   # |S2_mic1|
    }

    return Xc, S1c, S2c, feats, masks, mags


def main(dataset_dir="dataset"):
    ensure_dirs()

    mix_dir = os.path.join(dataset_dir, "mix")
    s1_dir = os.path.join(dataset_dir, "s1")
    s2_dir = os.path.join(dataset_dir, "s2")

    mix_files = sorted([f for f in os.listdir(mix_dir) if f.endswith(".wav")])
    if not mix_files:
        raise RuntimeError("No mix wav files found. Run generate_dataset.py first.")

    for idx, mf in enumerate(mix_files):
        # expected: mix_00000.wav
        sid = mf.replace("mix_", "").replace(".wav", "")
        mix_path = os.path.join(mix_dir, mf)
        s1_path = os.path.join(s1_dir, f"s1_{sid}.wav")
        s2_path = os.path.join(s2_dir, f"s2_{sid}.wav")

        x, sr = sf.read(mix_path, always_2d=True)
        s1, sr1 = sf.read(s1_path, always_2d=True)
        s2, sr2 = sf.read(s2_path, always_2d=True)

        if x.shape[1] != 2 or s1.shape[1] != 2 or s2.shape[1] != 2:
            raise RuntimeError(f"Expected stereo files. Problem with id={sid}")

        # STFT (2,F,T) complex
        X = stft_stereo(x)
        S1 = stft_stereo(s1)
        S2 = stft_stereo(s2)

        Xc, S1c, S2c, feats, masks, mags = compute_features_and_masks_and_mags(X, S1, S2)

        # Save complex arrays
        np.save(os.path.join(OUT_DIR, "X", f"{sid}.npy"), Xc)
        np.save(os.path.join(OUT_DIR, "S1", f"{sid}.npy"), S1c)
        np.save(os.path.join(OUT_DIR, "S2", f"{sid}.npy"), S2c)

        # Save features and masks
        np.save(os.path.join(OUT_DIR, "feats", f"{sid}.npy"), feats)
        np.save(os.path.join(OUT_DIR, "masks", f"{sid}.npy"), masks)

        # Save magnitudes for magnitude reconstruction loss
        np.savez(
            os.path.join(OUT_DIR, "mags", f"{sid}.npz"),
            mix_mag=mags["mix_mag"],
            s1_mag=mags["s1_mag"],
            s2_mag=mags["s2_mag"],
        )

        meta = {
            "id": sid,
            "orig_F": int(N_BINS),
            "target_F": int(TARGET_F),
            "target_T": int(TARGET_T),
            "mix_len_samples": int(x.shape[0]),
        }
        with open(os.path.join(OUT_DIR, "meta", f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx+1}/{len(mix_files)}")

    print("\nDone. Wrote STFT/feature/mask/mag files into stft_data/.")


if __name__ == "__main__":
    main("dataset")
