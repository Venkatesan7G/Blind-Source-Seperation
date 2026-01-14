import os
import json
import glob
import numpy as np
import soundfile as sf

from input_sound import load_audio_mono, crop_or_tile, TARGET_SR
from custom_room_mix import generate_stereo_bss_mix


def list_wavs(speech_dir: str):
    wavs = sorted(glob.glob(os.path.join(speech_dir, "*.wav")))
    wavs = [p for p in wavs if not os.path.basename(p).lower().endswith(".json")]
    if len(wavs) < 2:
        raise RuntimeError(f"Need at least 2 wav files in {speech_dir}, found {len(wavs)}")
    return wavs


def main(
    speech_dir="speech_pool",
    out_dir="dataset",
    num_mixes=1000,
    mix_seconds=4.0,
    seed=1234,
):
    rng = np.random.default_rng(seed)
    wav_paths = list_wavs(speech_dir)

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
            # pick two *different* utterances
            idx1 = int(rng.integers(0, len(wav_paths)))
            idx2 = int(rng.integers(0, len(wav_paths) - 1))
            if idx2 >= idx1:
                idx2 += 1

            p1 = wav_paths[idx1]
            p2 = wav_paths[idx2]

            s1, _ = load_audio_mono(p1, TARGET_SR)
            s2, _ = load_audio_mono(p2, TARGET_SR)

            # random crop/tile to fixed length
            s1_seg = crop_or_tile(s1, n_samples, rng)
            s2_seg = crop_or_tile(s2, n_samples, rng)

            # generate stereo room mixture
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
                "src1_file": p1.replace("\\", "/"),
                "src2_file": p2.replace("\\", "/"),
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
        num_mixes=1000,
        mix_seconds=4.0,
        seed=1234,
    )
