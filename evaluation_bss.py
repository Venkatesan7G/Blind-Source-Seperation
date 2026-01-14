import os
import json
import numpy as np
import soundfile as sf
from mir_eval.separation import bss_eval_sources

DATASET_DIR = "dataset"
SPLIT_PATH = "splits/split.json"
SEP_DIR = "separated_beamform"   # change to "separated" for mask-only

EPS = 1e-12

def load_ref_mic1(path):
    x, _ = sf.read(path, always_2d=True)
    return x[:, 0].astype(np.float32)

def load_est_mono(path):
    x, _ = sf.read(path, always_2d=True)
    # if mono saved, always_2d gives (N,1)
    return x[:, 0].astype(np.float32)

def eval_pit(ref1, ref2, est1, est2):
    """
    Evaluate both assignments:
      A: (est1->ref1, est2->ref2)
      B: (est2->ref1, est1->ref2)
    Pick the one with higher mean SDR.
    """
    L = min(len(ref1), len(ref2), len(est1), len(est2))
    ref = np.vstack([ref1[:L], ref2[:L]]).astype(np.float32)

    est_a = np.vstack([est1[:L], est2[:L]]).astype(np.float32)
    est_b = np.vstack([est2[:L], est1[:L]]).astype(np.float32)

    sdr_a, sir_a, sar_a, _ = bss_eval_sources(ref, est_a)
    sdr_b, sir_b, sar_b, _ = bss_eval_sources(ref, est_b)

    if float(np.mean(sdr_b)) > float(np.mean(sdr_a)):
        return sdr_b, sir_b, sar_b
    return sdr_a, sir_a, sar_a

def main():
    print("Using SEP_DIR:", SEP_DIR)
    split = json.load(open(SPLIT_PATH, "r", encoding="utf-8"))
    test_ids = split["test"]

    # quick existence check
    ex0 = test_ids[0]
    p = os.path.join(SEP_DIR, f"s1_hat_{ex0}.wav")
    print("Example est path:", p, "exists?", os.path.exists(p))

    SDRs, SIRs, SARs = [], [], []

    missing = 0
    for sid in test_ids:
        ref1 = load_ref_mic1(os.path.join(DATASET_DIR, "s1", f"s1_{sid}.wav"))
        ref2 = load_ref_mic1(os.path.join(DATASET_DIR, "s2", f"s2_{sid}.wav"))

        e1_path = os.path.join(SEP_DIR, f"s1_hat_{sid}.wav")
        e2_path = os.path.join(SEP_DIR, f"s2_hat_{sid}.wav")
        if not (os.path.exists(e1_path) and os.path.exists(e2_path)):
            missing += 1
            continue

        est1 = load_est_mono(e1_path)
        est2 = load_est_mono(e2_path)

        sdr, sir, sar = eval_pit(ref1, ref2, est1, est2)
        SDRs.append(sdr)
        SIRs.append(sir)
        SARs.append(sar)

    if missing > 0:
        print(f"WARNING: missing estimate pairs for {missing} items")

    SDRs = np.array(SDRs)  # (N,2)
    SIRs = np.array(SIRs)
    SARs = np.array(SARs)

    def mean_std(a):
        return float(np.mean(a)), float(np.std(a))

    sdr_m, sdr_s = mean_std(SDRs)
    sir_m, sir_s = mean_std(SIRs)
    sar_m, sar_s = mean_std(SARs)

    print("\n==== BSS Eval (mir_eval) on TEST set (mic1), PIT ====")
    print(f"SDR mean/std: {sdr_m:.2f} / {sdr_s:.2f} dB")
    print(f"SIR mean/std: {sir_m:.2f} / {sir_s:.2f} dB")
    print(f"SAR mean/std: {sar_m:.2f} / {sar_s:.2f} dB")

if __name__ == "__main__":
    main()
