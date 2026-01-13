import os
import json
import random

def main(stft_dir="stft_data", out_dir="splits", seed=1234):
    feats_dir = os.path.join(stft_dir, "feats")
    ids = sorted([f.replace(".npy", "") for f in os.listdir(feats_dir) if f.endswith(".npy")])
    if not ids:
        raise RuntimeError("No features found. Run feature_extraction.py first.")

    random.seed(seed)
    random.shuffle(ids)

    n = len(ids)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    split = {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "split.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    print(f"Total: {n}  train:{len(split['train'])}  val:{len(split['val'])}  test:{len(split['test'])}")
    print(f"Saved split file to: {out_path}")


if __name__ == "__main__":
    main()
