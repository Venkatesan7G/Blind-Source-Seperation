import os
import json
import numpy as np
import soundfile as sf
from mir_eval.separation import bss_eval_sources

DATASET_DIR = "dataset"
SPLIT_PATH = "splits/split.json"
SEP_DIR = "separated"

def load_ref_mic1(path):
    x, _ = sf.read(path, always_2d=True)
    return x[:, 0].astype(np.float32)

def load_est(path):
    x, _ = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]
    return x.astype(np.float32)

def main():
    split = json.load(open(SPLIT_PATH, "r", encoding="utf-8"))
    test_ids = split["test"]

    SDRs, SIRs, SARs = [], [], []

    for sid in test_ids:
        ref1 = load_ref_mic1(os.path.join(DATASET_DIR, "s1", f"s1_{sid}.wav"))
        ref2 = load_ref_mic1(os.path.join(DATASET_DIR, "s2", f"s2_{sid}.wav"))

        est1 = load_est(os.path.join(SEP_DIR, f"s1_hat_{sid}.wav"))
        est2 = load_est(os.path.join(SEP_DIR, f"s2_hat_{sid}.wav"))

        # align length
        L = min(len(ref1), len(ref2), len(est1), len(est2))
        ref = np.vstack([ref1[:L], ref2[:L]])
        est = np.vstack([est1[:L], est2[:L]])

        sdr, sir, sar, _ = bss_eval_sources(ref, est)

        SDRs.append(sdr)
        SIRs.append(sir)
        SARs.append(sar)

    SDRs = np.array(SDRs)  # (N,2)
    SIRs = np.array(SIRs)
    SARs = np.array(SARs)

    # Report mean/std across both sources
    def mean_std(a):
        return float(np.mean(a)), float(np.std(a))

    sdr_m, sdr_s = mean_std(SDRs)
    sir_m, sir_s = mean_std(SIRs)
    sar_m, sar_s = mean_std(SARs)

    print("\n==== BSS Eval (mir_eval) on TEST set (mic1) ====")
    print(f"SDR mean/std: {sdr_m:.2f} / {sdr_s:.2f} dB")
    print(f"SIR mean/std: {sir_m:.2f} / {sir_s:.2f} dB")
    print(f"SAR mean/std: {sar_m:.2f} / {sar_s:.2f} dB")

if __name__ == "__main__":
    main()
