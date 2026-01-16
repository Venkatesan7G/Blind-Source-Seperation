import os
import json
import numpy as np
import soundfile as sf

from stft_util import stft_stereo, crop_F, pad_or_trim_T

DATASET_DIR = "dataset"
SPLIT_PATH = "splits/split.json"
OUT_DIR = "stft_data"

TARGET_F = 256
TARGET_T = 620

def load_stereo(path: str):
    x, sr = sf.read(path, always_2d=True)
    if x.shape[1] != 2:
        raise ValueError(f"Expected stereo file at {path}, got shape {x.shape}")
    return x.astype(np.float32), int(sr)

def main():
    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"Missing {SPLIT_PATH}. Run: python split_data.py")

    split = json.load(open(SPLIT_PATH, "r", encoding="utf-8"))
    all_ids = split["train"] + split["val"] + split["test"]

    X_dir = os.path.join(OUT_DIR, "X")
    S_dir = os.path.join(OUT_DIR, "S")
    meta_dir = os.path.join(OUT_DIR, "meta")
    os.makedirs(X_dir, exist_ok=True)
    os.makedirs(S_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    print(f"Extracting STFT for {len(all_ids)} items ...")

    for k, sid in enumerate(all_ids, start=1):
        mix_path = os.path.join(DATASET_DIR, "mix", f"mix_{sid}.wav")
        s1_path  = os.path.join(DATASET_DIR, "s1",  f"s1_{sid}.wav")
        s2_path  = os.path.join(DATASET_DIR, "s2",  f"s2_{sid}.wav")

        mix, sr = load_stereo(mix_path)
        s1, sr1 = load_stereo(s1_path)
        s2, sr2 = load_stereo(s2_path)
        if not (sr == sr1 == sr2):
            raise ValueError(f"SR mismatch for id={sid}: {sr},{sr1},{sr2}")

        mix_len = mix.shape[0]

        # STFT: (2, Ffull, Tvar) complex
        X = stft_stereo(mix)
        S1 = stft_stereo(s1)[0]  # mic1 ref image (Ffull,Tvar)
        S2 = stft_stereo(s2)[0]

        # Crop/pad to fixed sizes
        Xc = crop_F(X, TARGET_F)
        Xc = pad_or_trim_T(Xc, TARGET_T)

        S1c = crop_F(np.stack([S1, S1], axis=0), TARGET_F)[0]
        S1c = pad_or_trim_T(np.stack([S1c, S1c], axis=0), TARGET_T)[0]

        S2c = crop_F(np.stack([S2, S2], axis=0), TARGET_F)[0]
        S2c = pad_or_trim_T(np.stack([S2c, S2c], axis=0), TARGET_T)[0]

        np.save(os.path.join(X_dir, f"{sid}.npy"), Xc.astype(np.complex64))
        np.savez(os.path.join(S_dir, f"{sid}.npz"),
                 S1=S1c.astype(np.complex64),
                 S2=S2c.astype(np.complex64))

        meta = {"id": sid, "sr": sr, "mix_len_samples": int(mix_len)}
        with open(os.path.join(meta_dir, f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)

        if k % 50 == 0 or k == len(all_ids):
            print(f"  done {k}/{len(all_ids)}")

    print("Done. Wrote STFT cache to:", OUT_DIR)

if __name__ == "__main__":
    main()
