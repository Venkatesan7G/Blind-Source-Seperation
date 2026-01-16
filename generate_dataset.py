import os
import json
import glob
import numpy as np
import soundfile as sf

from input_sound import load_audio_mono, crop_or_tile, TARGET_SR
from custom_room_mix import generate_stereo_bss_mix


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def apply_snr(s1: np.ndarray, s2: np.ndarray, snr_db: float):
    """
    Scale s2 so that RMS(s1) / RMS(s2_scaled) = 10^(snr_db/20)
    Positive snr_db means s1 louder than s2.
    """
    r1 = rms(s1)
    r2 = rms(s2)
    target_ratio = 10.0 ** (snr_db / 20.0)
    s2_scaled = s2 * (r1 / (r2 * target_ratio + 1e-12))
    return s2_scaled.astype(np.float32)


def load_speech_pool(speech_dir: str):
    """
    Reads speech_pool/meta.jsonl if exists; otherwise just lists .wav files.
    Returns list of dicts: {"path":..., "speaker_id":...}
    """
    meta_path = os.path.join(speech_dir, "meta.jsonl")
    items = []

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                items.append({"path": j["path"], "speaker_id": str(j.get("speaker_id", "unknown"))})
    else:
        wavs = sorted(glob.glob(os.path.join(speech_dir, "*.wav")))
        if not wavs:
            raise RuntimeError(f"No wavs found in {speech_dir}")
        # fallback: speaker_id unknown for all
        items = [{"path": p.replace("\\", "/"), "speaker_id": "unknown"} for p in wavs]

    if len(items) < 2:
        raise RuntimeError("Need at least 2 speech files.")
    return items


def build_speaker_index(items):
    spk2items = {}
    for it in items:
        spk = it["speaker_id"]
        spk2items.setdefault(spk, []).append(it)
    # keep speakers with at least 1 file
    speakers = [s for s in spk2items.keys() if len(spk2items[s]) >= 1]
    return spk2items, speakers


def main(
    speech_dir="speech_pool",
    out_dir="dataset",
    num_mixes=3000,
    mix_seconds=4.0,
    snr_db_range=(-5.0, 5.0),
    seed=1234,
):
    rng = np.random.default_rng(seed)

    items = load_speech_pool(speech_dir)
    spk2items, speakers = build_speaker_index(items)

    if len(speakers) < 2:
        raise RuntimeError(
            f"Need at least 2 different speakers in meta.jsonl. Found speakers={len(speakers)}."
        )

    # Output dirs
    mix_dir = os.path.join(out_dir, "mix")
    s1_dir = os.path.join(out_dir, "s1")
    s2_dir = os.path.join(out_dir, "s2")
    os.makedirs(mix_dir, exist_ok=True)
    os.makedirs(s1_dir, exist_ok=True)
    os.makedirs(s2_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.jsonl")

    n_samples = int(mix_seconds * TARGET_SR)

    with open(meta_path, "w", encoding="utf-8") as mf:
        for i in range(num_mixes):
            # pick two different speakers
            spk1, spk2 = rng.choice(speakers, size=2, replace=False)
            it1 = spk2items[spk1][int(rng.integers(0, len(spk2items[spk1])))]
            it2 = spk2items[spk2][int(rng.integers(0, len(spk2items[spk2])))]
            p1 = it1["path"]
            p2 = it2["path"]

            s1, _ = load_audio_mono(p1, TARGET_SR)
            s2, _ = load_audio_mono(p2, TARGET_SR)

            s1_seg = crop_or_tile(s1, n_samples, rng)
            s2_seg = crop_or_tile(s2, n_samples, rng)

            # random SNR
            snr_db = float(rng.uniform(snr_db_range[0], snr_db_range[1]))
            s2_seg = apply_snr(s1_seg, s2_seg, snr_db)

            # stereo room mix
            mix, s1_img, s2_img, meta = generate_stereo_bss_mix(
                s1_seg,
                s2_seg,
                sr=TARGET_SR,
                seed=int(rng.integers(0, 2**31 - 1)),
            )

            mix_path = os.path.join(mix_dir, f"mix_{i:05d}.wav")
            s1_path = os.path.join(s1_dir, f"s1_{i:05d}.wav")
            s2_path = os.path.join(s2_dir, f"s2_{i:05d}.wav")

            sf.write(mix_path, mix, TARGET_SR, subtype="FLOAT")
            sf.write(s1_path, s1_img, TARGET_SR, subtype="FLOAT")
            sf.write(s2_path, s2_img, TARGET_SR, subtype="FLOAT")

            meta_out = {
                "id": f"{i:05d}",
                "mix_path": mix_path.replace("\\", "/"),
                "s1_path": s1_path.replace("\\", "/"),
                "s2_path": s2_path.replace("\\", "/"),
                "mix_seconds": mix_seconds,
                "snr_db_s1_over_s2": snr_db,
                "src1_file": p1.replace("\\", "/"),
                "src2_file": p2.replace("\\", "/"),
                "src1_speaker": str(spk1),
                "src2_speaker": str(spk2),
                **meta,
            }
            mf.write(json.dumps(meta_out) + "\n")

            if (i + 1) % 50 == 0:
                print(f"Generated {i+1}/{num_mixes}")

    print(f"\nDone. Dataset written to: {out_dir}")
    print(f"Mixes: {mix_dir}")
    print(f"Sources: {s1_dir}, {s2_dir}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main(
        speech_dir="speech_pool",
        out_dir="dataset",
        num_mixes=3000,      # start 2000; later try 3000-5000
        mix_seconds=4.0,
        snr_db_range=(-5.0, 5.0),
        seed=1234,
    )
