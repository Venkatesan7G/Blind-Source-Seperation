import os
import json
import numpy as np
import soundfile as sf
import torch

from torch_model import TFMaskUNet
from stft_util import istft_1d  # make sure your file is stft_util.py
# if your file is stft_utils.py then import from that instead


STFT_DIR = "stft_data"
SPLIT_PATH = "splits/split.json"
CKPT_PATH = "checkpoints_torch/best.pt"
OUT_DIR = "separated"
META_DIR = os.path.join(STFT_DIR, "meta")
FEATS_DIR = os.path.join(STFT_DIR, "feats")
X_DIR = os.path.join(STFT_DIR, "X")

SR = 16000
EPS = 1e-8


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    split = json.load(open(SPLIT_PATH, "r", encoding="utf-8"))
    test_ids = split["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    in_ch = ckpt["in_ch"]
    base = ckpt["base"]

    model = TFMaskUNet(in_ch=in_ch, base=base, n_src=2).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    for k, sid in enumerate(test_ids):
        feats = np.load(os.path.join(FEATS_DIR, f"{sid}.npy")).astype(np.float32)  # (F,T,5)
        X = np.load(os.path.join(X_DIR, f"{sid}.npy"))  # complex (2,F,T) cropped/padded

        # torch input: (1,5,F,T)
        x = torch.from_numpy(feats).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            masks = model(x)[0].cpu().numpy().astype(np.float32)  # (2,F,T)

        # use mic1 mixture STFT
        X1 = X[0]  # (F,T) complex
        S1_hat = (masks[0] * X1).astype(np.complex64)
        S2_hat = (masks[1] * X1).astype(np.complex64)

        meta = json.load(open(os.path.join(META_DIR, f"{sid}.json"), "r", encoding="utf-8"))
        length = int(meta["mix_len_samples"])

        s1_t = istft_1d(S1_hat, length=length)
        s2_t = istft_1d(S2_hat, length=length)

        mx = max(np.max(np.abs(s1_t)) + 1e-12, np.max(np.abs(s2_t)) + 1e-12)
        if mx > 0.99:
            s1_t = 0.99 * s1_t / mx
            s2_t = 0.99 * s2_t / mx

        sf.write(os.path.join(OUT_DIR, f"s1_hat_{sid}.wav"), s1_t, SR, subtype="FLOAT")
        sf.write(os.path.join(OUT_DIR, f"s2_hat_{sid}.wav"), s2_t, SR, subtype="FLOAT")

        if (k + 1) % 25 == 0:
            print(f"Separated {k+1}/{len(test_ids)}")

    print("Done. Wrote:", OUT_DIR)


if __name__ == "__main__":
    main()
