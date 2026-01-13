import os
import json
import numpy as np
import soundfile as sf

from input_sound import load_audio_mono, crop_or_tile, TARGET_SR
from custom_room_mix import generate_stereo_bss_mix


def main(
    male_path="male.wav",
    female_path="female.wav",
    out_dir="dataset",
    num_mixes=300,
    mix_seconds=4.0,
    seed=1234,
):
    rng = np.random.default_rng(seed)

    # Load dry sources
    male, sr_m = load_audio_mono(male_path, TARGET_SR)
    female, sr_f = load_audio_mono(female_path, TARGET_SR)
    assert sr_m == TARGET_SR and sr_f == TARGET_SR

    sources = [male, female]

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
            # Always keep identity consistent
            s1 = male
            s2 = female

            # Random crop/tile to get different content each mix
            s1_seg = crop_or_tile(s1, n_samples, rng)
            s2_seg = crop_or_tile(s2, n_samples, rng)

            mix, s1_img, s2_img, meta = generate_stereo_bss_mix(
                s1_seg,
                s2_seg,
                sr=TARGET_SR,
                seed=int(rng.integers(0, 2**31 - 1)),
    )

            # Save WAVs (float32)
            mix_path = os.path.join(mix_dir, f"mix_{i:05d}.wav")
            s1_path = os.path.join(s1_dir, f"s1_{i:05d}.wav")
            s2_path = os.path.join(s2_dir, f"s2_{i:05d}.wav")

            sf.write(mix_path, mix, TARGET_SR, subtype="FLOAT")
            sf.write(s1_path, s1_img, TARGET_SR, subtype="FLOAT")
            sf.write(s2_path, s2_img, TARGET_SR, subtype="FLOAT")

            # meta line
            meta_out = {
                "id": f"{i:05d}",
                "mix_path": mix_path.replace("\\", "/"),
                "s1_path": s1_path.replace("\\", "/"),
                "s2_path": s2_path.replace("\\", "/"),
                "mix_seconds": mix_seconds,
                **meta,
            }
            mf.write(json.dumps(meta_out) + "\n")

            if (i + 1) % 25 == 0:
                print(f"Generated {i+1}/{num_mixes}")

    print(f"\nDone. Dataset written to: {out_dir}")
    print(f"Mixes: {mix_dir}")
    print(f"Sources: {s1_dir}, {s2_dir}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    # Edit male/female paths to your actual files
    main(
        male_path="male.wav",
        female_path="female.wav",
        out_dir="dataset",
        num_mixes=300,      # good starting point; later you can raise to 500+
        mix_seconds=4.0,    # fixed-length examples
        seed=1234,
    )
