import os
import json
import numpy as np
import soundfile as sf
import tensorflow as tf

from stft_util import istft_1d, crop_F, pad_or_trim_T

STFT_DIR = "stft_data"
SPLIT_PATH = "splits/split.json"
MODEL_PATH = "checkpoints/best.keras"

OUT_DIR = "separated"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-8

def main():
    split = json.load(open(SPLIT_PATH, "r", encoding="utf-8"))
    test_ids = split["test"]

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    feats_dir = os.path.join(STFT_DIR, "feats")
    X_dir = os.path.join(STFT_DIR, "X")
    meta_dir = os.path.join(STFT_DIR, "meta")

    for k, sid in enumerate(test_ids):
        feats = np.load(os.path.join(feats_dir, f"{sid}.npy")).astype(np.float32)  # (F,T,5)
        X = np.load(os.path.join(X_dir, f"{sid}.npy"))  # complex (2,F,T)

        # model expects (B,F,T,5)
        masks = model.predict(feats[None, ...], verbose=0)[0].astype(np.float32)   # (F,T,2)

        # enforce sum-to-1 again (safety)
        denom = np.sum(masks, axis=-1, keepdims=True) + EPS
        masks = masks / denom

        # Use mic1 mixture STFT
        X1 = X[0]  # (F,T) complex (already cropped/padded in feature_extraction)
        # Apply masks
        S1_hat = (masks[..., 0] * X1).astype(np.complex64)
        S2_hat = (masks[..., 1] * X1).astype(np.complex64)

        # Get target length (samples) for ISTFT
        meta_path = os.path.join(meta_dir, f"{sid}.json")
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        length = int(meta["mix_len_samples"])

        s1_t = istft_1d(S1_hat, length=length)
        s2_t = istft_1d(S2_hat, length=length)

        # Normalize safely
        mx = max(np.max(np.abs(s1_t)) + 1e-12, np.max(np.abs(s2_t)) + 1e-12)
        if mx > 0.99:
            s1_t = 0.99 * s1_t / mx
            s2_t = 0.99 * s2_t / mx

        sf.write(os.path.join(OUT_DIR, f"s1_hat_{sid}.wav"), s1_t, 16000, subtype="FLOAT")
        sf.write(os.path.join(OUT_DIR, f"s2_hat_{sid}.wav"), s2_t, 16000, subtype="FLOAT")

        if (k + 1) % 25 == 0:
            print(f"Separated {k+1}/{len(test_ids)}")

    print(f"Done. Wrote separated audio into: {OUT_DIR}")

if __name__ == "__main__":
    main()
