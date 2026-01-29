import os
import json
import random

DATASET_DIR = "dataset"
OUT_DIR = "splits"

def main(dataset_dir=DATASET_DIR, out_dir=OUT_DIR, seed=1234):
    mix_dir = os.path.join(dataset_dir, "mix")
    if not os.path.isdir(mix_dir):
        raise FileNotFoundError(f"Missing folder: {mix_dir}")

    ids = []
    for fn in os.listdir(mix_dir):
        if fn.startswith("mix_") and fn.endswith(".wav"):
            sid = fn[len("mix_"):-len(".wav")]
            ids.append(sid)

    ids = sorted(ids)
    if not ids:
        raise RuntimeError(f"No mix files found in {mix_dir}")

    random.seed(seed)
    random.shuffle(ids)

    n = len(ids)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

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
