import os
import json
import numpy as np
import soundfile as sf

from stft_util import stft_stereo, crop_F, pad_or_trim_T, N_BINS

# Output config
OUT_DIR = "stft_data"
TARGET_F = 256          # crop 257 -> 256 to simplify pooling
TARGET_T = 620          # fixed frames (like your old code)
EPS = 1e-8


def ensure_dirs():
    for sub in ["X", "S1", "S2", "feats", "masks", "meta"]:
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)


def compute_features_and_masks(X: np.ndarray, S1: np.ndarray, S2: np.ndarray):
    """
    X, S1, S2: complex STFT (2,F,T)
    Returns:
      feats: (F,T,5)
      masks: (F,T,2)  ideal ratio masks at mic1
    """
    # Crop F first
    Xc = crop_F(X, TARGET_F)
    S1c = crop_F(S1, TARGET_F)
    S2c = crop_F(S2, TARGET_F)

    # Pad/trim T
    Xc = pad_or_trim_T(Xc, TARGET_T)
    S1c = pad_or_trim_T(S1c, TARGET_T)
    S2c = pad_or_trim_T(S2c, TARGET_T)

    # Mixture features
    X1 = Xc[0]  # (F,T)
    X2 = Xc[1]

    mag1 = np.abs(X1)
    mag2 = np.abs(X2)
    ipd = np.angle(X1) - np.angle(X2)
    ild = np.log(mag1 + EPS) - np.log(mag2 + EPS)

    feats = np.stack([mag1, mag2, np.cos(ipd), np.sin(ipd), ild], axis=-1).astype(np.float32)  # (F,T,5)

    # Ideal Ratio Masks (mic1 images)
    S1m1 = np.abs(S1c[0])
    S2m1 = np.abs(S2c[0])
    denom = S1m1 + S2m1 + EPS
    M1 = (S1m1 / denom).astype(np.float32)
    M2 = (S2m1 / denom).astype(np.float32)
    masks = np.stack([M1, M2], axis=-1)  # (F,T,2)

    return Xc, S1c, S2c, feats, masks


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

        # STFT (2,F,T)
        X = stft_stereo(x)
        S1 = stft_stereo(s1)
        S2 = stft_stereo(s2)

        Xc, S1c, S2c, feats, masks = compute_features_and_masks(X, S1, S2)

        # Save
        np.save(os.path.join(OUT_DIR, "X", f"{sid}.npy"), Xc)
        np.save(os.path.join(OUT_DIR, "S1", f"{sid}.npy"), S1c)
        np.save(os.path.join(OUT_DIR, "S2", f"{sid}.npy"), S2c)
        np.save(os.path.join(OUT_DIR, "feats", f"{sid}.npy"), feats)
        np.save(os.path.join(OUT_DIR, "masks", f"{sid}.npy"), masks)

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

    print("\nDone. Wrote STFT/feature/mask files into stft_data/.")


if __name__ == "__main__":
    main("dataset")
