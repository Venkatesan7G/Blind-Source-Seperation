import os
import json
import numpy as np
import tensorflow as tf

class BSSMaskGenerator(tf.keras.utils.Sequence):
    """
    Loads:
      X features: stft_data/feats/{id}.npy   shape (F,T,5)
      Y masks:    stft_data/masks/{id}.npy  shape (F,T,2)

    Returns:
      X_batch: (B,F,T,5)
      Y_batch: (B,F,T,2)
    """
    def __init__(
        self,
        split_name: str,
        split_json_path: str = "splits/split.json",
        stft_dir: str = "stft_data",
        batch_size: int = 8,
        shuffle: bool = True,
        seed: int = 1234,
    ):
        self.split_name = split_name
        self.split = json.load(open(split_json_path, "r", encoding="utf-8"))
        self.ids = list(self.split[split_name])
        self.stft_dir = stft_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        self.feats_dir = os.path.join(stft_dir, "feats")
        self.masks_dir = os.path.join(stft_dir, "masks")

        self.indexes = np.arange(len(self.ids))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.ids) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_ids = [self.ids[i] for i in batch_idx]

        X_list, Y_list = [], []
        for sid in batch_ids:
            X = np.load(os.path.join(self.feats_dir, f"{sid}.npy")).astype(np.float32)   # (F,T,5)
            Y = np.load(os.path.join(self.masks_dir, f"{sid}.npy")).astype(np.float32)  # (F,T,2)
            X_list.append(X)
            Y_list.append(Y)

        Xb = np.stack(X_list, axis=0)  # (B,F,T,5)
        Yb = np.stack(Y_list, axis=0)  # (B,F,T,2)
        return Xb, Yb
