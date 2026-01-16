import os
import json
import random

DATASET_DIR = "dataset"
OUT_DIR = "splits"

def main(dataset_dir=DATASET_DIR, out_dir=OUT_DIR, seed=1234, train_p=0.70, val_p=0.15):
    mix_dir = os.path.join(dataset_dir, "mix")
    if not os.path.exists(mix_dir):
        raise FileNotFoundError(f"Missing {mix_dir}. Generate dataset first.")

    # ids from mix_XXXXX.wav
    ids = []
    for f in os.listdir(mix_dir):
        if f.startswith("mix_") and f.endswith(".wav"):
            sid = f[len("mix_"):-len(".wav")]
            if len(sid) == 5 and sid.isdigit():
                ids.append(sid)

    ids = sorted(ids)
    if not ids:
        raise RuntimeError(f"No mixture wavs found in {mix_dir}")

    random.seed(seed)
    random.shuffle(ids)

    n = len(ids)
    n_train = int(train_p * n)
    n_val = int(val_p * n)
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
