# generate_dataset_IMPROVED.py - BETTER DATASET GENERATION
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
        items = [{"path": p.replace("\\", "/"), "speaker_id": "unknown"} for p in wavs]

    if len(items) < 2:
        raise RuntimeError("Need at least 2 speech files.")
    return items


def build_speaker_index(items):
    """Build index of utterances per speaker"""
    spk2items = {}
    for it in items:
        spk = it["speaker_id"]
        spk2items.setdefault(spk, []).append(it)
    speakers = [s for s in spk2items.keys() if len(spk2items[s]) >= 1]
    return spk2items, speakers


def main(
    speech_dir="speech_pool",
    out_dir="dataset",
    num_mixes=6000,          # ⭐ INCREASED from 3000 to 6000
    mix_seconds=4.5,         # ⭐ INCREASED from 4.0 to 4.5 (more context)
    snr_db_range=(-5.0, 8.0), # ⭐ WIDER range from (-5, 5) to (-5, 8)
    seed=1234,
):
    """
    Generate stereo mixtures for BSS training
    
    ⭐ KEY IMPROVEMENTS:
    1. More mixtures (6000 instead of 3000)
    2. Longer segments (4.5s instead of 4.0s)
    3. Wider SNR range for more diversity
    4. Better speaker diversity handling
    """
    rng = np.random.default_rng(seed)

    items = load_speech_pool(speech_dir)
    spk2items, speakers = build_speaker_index(items)

    print("=" * 80)
    print("DATASET GENERATION - IMPROVED")
    print("=" * 80)
    print(f"Speech pool directory: {speech_dir}")
    print(f"Total utterances: {len(items)}")
    print(f"Unique speakers: {len(speakers)}")
    print(f"Target mixtures: {num_mixes}")
    print(f"Mix duration: {mix_seconds}s")
    print(f"SNR range: {snr_db_range[0]} to {snr_db_range[1]} dB")
    print("=" * 80 + "\n")

    if len(speakers) < 2:
        raise RuntimeError(
            f"❌ Need at least 2 different speakers. Found {len(speakers)}.\n"
            f"Make sure your LibriSpeech download has speaker_id metadata!"
        )
    
    # ⭐ RECOMMENDATION: Warn if too few speakers
    if len(speakers) < 20:
        print(f"⚠️  WARNING: Only {len(speakers)} speakers found.")
        print(f"   For best results, use 50+ speakers from LibriSpeech.")
        print(f"   Consider downloading more with fetch_libre_stream.py\n")

    # Output directories
    mix_dir = os.path.join(out_dir, "mix")
    s1_dir = os.path.join(out_dir, "s1")
    s2_dir = os.path.join(out_dir, "s2")
    os.makedirs(mix_dir, exist_ok=True)
    os.makedirs(s1_dir, exist_ok=True)
    os.makedirs(s2_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.jsonl")

    n_samples = int(mix_seconds * TARGET_SR)

    # Statistics tracking
    speaker_pair_counts = {}
    snr_values = []
    room_rt60_values = []
    
    with open(meta_path, "w", encoding="utf-8") as mf:
        for i in range(num_mixes):
            # ⭐ IMPROVED: Pick different speakers (critical for generalization!)
            spk1, spk2 = rng.choice(speakers, size=2, replace=False)
            
            # Track speaker pair diversity
            pair_key = tuple(sorted([spk1, spk2]))
            speaker_pair_counts[pair_key] = speaker_pair_counts.get(pair_key, 0) + 1
            
            # Random utterances from each speaker
            it1 = spk2items[spk1][int(rng.integers(0, len(spk2items[spk1])))]
            it2 = spk2items[spk2][int(rng.integers(0, len(spk2items[spk2])))]
            p1 = it1["path"]
            p2 = it2["path"]

            # Load and prepare segments
            s1, _ = load_audio_mono(p1, TARGET_SR)
            s2, _ = load_audio_mono(p2, TARGET_SR)

            s1_seg = crop_or_tile(s1, n_samples, rng)
            s2_seg = crop_or_tile(s2, n_samples, rng)

            # Random SNR
            snr_db = float(rng.uniform(snr_db_range[0], snr_db_range[1]))
            snr_values.append(snr_db)
            s2_seg = apply_snr(s1_seg, s2_seg, snr_db)

            # Generate stereo room mixture
            mix, s1_img, s2_img, meta = generate_stereo_bss_mix(
                s1_seg,
                s2_seg,
                sr=TARGET_SR,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            
            room_rt60_values.append(meta["rt60"])

            # Save files
            mix_path = os.path.join(mix_dir, f"mix_{i:05d}.wav")
            s1_path = os.path.join(s1_dir, f"s1_{i:05d}.wav")
            s2_path = os.path.join(s2_dir, f"s2_{i:05d}.wav")

            sf.write(mix_path, mix, TARGET_SR, subtype="FLOAT")
            sf.write(s1_path, s1_img, TARGET_SR, subtype="FLOAT")
            sf.write(s2_path, s2_img, TARGET_SR, subtype="FLOAT")

            # Write metadata
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

            # Progress reporting
            if (i + 1) % 100 == 0 or (i + 1) == num_mixes:
                pct = 100 * (i + 1) / num_mixes
                print(f"Generated {i+1:5d}/{num_mixes} ({pct:5.1f}%)")

    # Final statistics
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {out_dir}")
    print(f"Total mixtures: {num_mixes}")
    print(f"\nDataset statistics:")
    print(f"  Unique speaker pairs: {len(speaker_pair_counts)}")
    print(f"  SNR - min: {min(snr_values):.1f} dB, max: {max(snr_values):.1f} dB, mean: {np.mean(snr_values):.1f} dB")
    print(f"  RT60 - min: {min(room_rt60_values):.3f} s, max: {max(room_rt60_values):.3f} s, mean: {np.mean(room_rt60_values):.3f} s")
    
    # Check speaker diversity
    max_pair_count = max(speaker_pair_counts.values())
    if max_pair_count > num_mixes / len(speakers):
        print(f"\n⚠️  WARNING: Some speaker pairs are over-represented (max {max_pair_count} times)")
        print(f"   This may reduce generalization. Consider adding more speakers.")
    
    print("=" * 80)


if __name__ == "__main__":
    # ⭐ RECOMMENDED SETTINGS FOR 6000 MIXTURES
    main(
        speech_dir="speech_pool",
        out_dir="dataset",
        num_mixes=6000,          # Target 6000 mixtures
        mix_seconds=4.5,         # Slightly longer for more context
        snr_db_range=(-5.0, 8.0), # Wider range for diversity
        seed=1234,
    )