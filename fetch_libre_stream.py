# fetch_libre_stream_IMPROVED.py - BETTER LIBRISPEECH DOWNLOAD
import os
import random
import argparse
import numpy as np
import soundfile as sf
import io
from collections import defaultdict

from datasets import load_dataset, Audio


def main(
    out_dir="speech_pool",
    num_files=800,          # ⭐ INCREASED from 200 to 800
    seed=1234,
    split="train.clean.100", # ⭐ CHANGED to larger split
    min_duration=3.0,       # ⭐ NEW: Filter short utterances
    max_files_per_speaker=20 # ⭐ NEW: Limit per speaker for diversity
):
    """
    Download LibriSpeech utterances with better diversity
    
    ⭐ KEY IMPROVEMENTS:
    1. Download from larger split (train.clean.100 instead of test.clean)
    2. More files (800 instead of 200)
    3. Ensure speaker diversity (max 20 files per speaker)
    4. Filter out very short utterances
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("LIBRISPEECH DOWNLOAD - IMPROVED")
    print("=" * 80)
    print(f"Output directory: {out_dir}")
    print(f"Target files: {num_files}")
    print(f"Split: {split}")
    print(f"Min duration: {min_duration}s")
    print(f"Max per speaker: {max_files_per_speaker}")
    print("=" * 80 + "\n")

    # Load dataset in streaming mode
    print("Loading LibriSpeech dataset (streaming)...")
    ds = load_dataset("openslr/librispeech_asr", split=split, streaming=True)
    
    # Don't decode audio (avoid ffmpeg issues)
    ds = ds.cast_column("audio", Audio(decode=False))

    meta_path = os.path.join(out_dir, "meta.jsonl")
    written = 0
    skipped_short = 0
    skipped_speaker_limit = 0
    
    # Track files per speaker for diversity
    speaker_counts = defaultdict(int)
    speaker_ids_seen = set()

    with open(meta_path, "w", encoding="utf-8") as mf:
        for ex in ds:
            if written >= num_files:
                break
                
            audio = ex["audio"]
            audio_bytes = audio.get("bytes", None)
            if audio_bytes is None:
                continue

            # Decode audio
            try:
                wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            except Exception as e:
                print(f"Warning: Failed to decode audio: {e}")
                continue
                
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)

            # ⭐ NEW: Skip very short utterances
            duration = len(wav) / sr
            if duration < min_duration:
                skipped_short += 1
                continue

            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
                sr = 16000

            # Normalize
            peak = float(np.max(np.abs(wav)) + 1e-12)
            if peak > 0.99:
                wav = 0.99 * wav / peak

            # Get metadata
            utt_id = ex.get("id", f"{written}")
            spk_id = ex.get("speaker_id", "unknown")
            
            # ⭐ NEW: Enforce speaker diversity
            if speaker_counts[spk_id] >= max_files_per_speaker:
                skipped_speaker_limit += 1
                continue
            
            speaker_counts[spk_id] += 1
            speaker_ids_seen.add(spk_id)

            # Save file
            out_path = os.path.join(out_dir, f"ls_{written:05d}_spk{spk_id}_{utt_id}.wav")
            sf.write(out_path, wav, sr, subtype="FLOAT")

            # Write metadata
            mf.write(
                '{"path":"%s","speaker_id":"%s","utt_id":"%s","split":"%s","duration_s":%.2f}\n'
                % (out_path.replace("\\", "/"), str(spk_id), str(utt_id), split, duration)
            )

            written += 1
            if written % 50 == 0:
                print(f"Saved {written}/{num_files} files ({len(speaker_ids_seen)} speakers)")

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"Saved {written} wav files to: {out_dir}")
    print(f"Unique speakers: {len(speaker_ids_seen)}")
    print(f"Metadata file: {meta_path}")
    
    if skipped_short > 0:
        print(f"\nSkipped {skipped_short} utterances (too short, < {min_duration}s)")
    if skipped_speaker_limit > 0:
        print(f"Skipped {skipped_speaker_limit} utterances (speaker limit reached)")
    
    # Speaker diversity check
    avg_per_speaker = written / len(speaker_ids_seen) if speaker_ids_seen else 0
    print(f"\nAverage files per speaker: {avg_per_speaker:.1f}")
    
    if len(speaker_ids_seen) < 30:
        print(f"\n⚠️  WARNING: Only {len(speaker_ids_seen)} speakers!")
        print(f"   For best results, aim for 50+ speakers.")
        print(f"   Try increasing num_files or using 'train.clean.360' split.")
    
    print("=" * 80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download LibriSpeech with better diversity")
    ap.add_argument("--out_dir", default="speech_pool", help="Output directory")
    ap.add_argument("--num_files", type=int, default=800, help="Number of files to download")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed")
    ap.add_argument("--split", default="train.clean.100", 
                    help="LibriSpeech split (train.clean.100, train.clean.360, dev.clean, test.clean)")
    ap.add_argument("--min_duration", type=float, default=3.0,
                    help="Minimum utterance duration in seconds")
    ap.add_argument("--max_per_speaker", type=int, default=20,
                    help="Maximum files per speaker (for diversity)")
    
    args = ap.parse_args()
    
    main(
        out_dir=args.out_dir,
        num_files=args.num_files,
        seed=args.seed,
        split=args.split,
        min_duration=args.min_duration,
        max_files_per_speaker=args.max_per_speaker
    )