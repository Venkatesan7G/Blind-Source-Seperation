import argparse
import os
import subprocess
import sys
from datetime import datetime


PIPELINE = [
    ("Generate dataset", "generate_dataset.py"),
    ("Extract STFT features + masks", "feature_extraction.py"),
    ("Create train/val/test split", "split_data.py"),
    ("Train mask model", "train.py"),
    ("Separate test set", "separate.py"),
    ("Evaluate SDR/SIR/SAR", "evaluate_bss.py"),
]


def run_step(step_name: str, script: str, python_exe: str, dry_run: bool = False) -> None:
    if not os.path.exists(script):
        raise FileNotFoundError(f"Required script not found: {script}")

    cmd = [python_exe, script]

    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STEP: {step_name}")
    print("CMD:", " ".join(cmd))
    print("=" * 80)

    if dry_run:
        print("(dry-run) skipping execution")
        return

    # Run and stream output live
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Step failed: {step_name} ({script}) exited with code {ret}")

    print(f"\n✅ Completed: {step_name}")


def check_expected_files() -> None:
    """
    Quick sanity checks so you fail fast with a helpful message.
    """
    # Dataset check
    if not os.path.exists("dataset/mix"):
        print("⚠️  dataset/mix not found (dataset may not be generated yet).")
    # STFT/feature check
    if not os.path.exists("stft_data/feats"):
        print("⚠️  stft_data/feats not found (features may not be extracted yet).")
    # Split check
    if not os.path.exists("splits/split.json"):
        print("⚠️  splits/split.json not found (split_data may not have run yet).")
    # Model check
    if not os.path.exists("checkpoints/best.keras"):
        print("⚠️  checkpoints/best.keras not found (training may not have run yet).")


def main():
    parser = argparse.ArgumentParser(
        description="Run the stereo TF-domain BSS pipeline end-to-end."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current interpreter)",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        help="Start from this step number (1-based). Default: 1",
    )
    parser.add_argument(
        "--to-step",
        type=int,
        default=len(PIPELINE),
        help=f"Run up to this step number (1-based). Default: {len(PIPELINE)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands but do not execute",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only run quick filesystem checks, do not execute pipeline",
    )

    args = parser.parse_args()

    if args.check:
        check_expected_files()
        return

    if args.from_step < 1 or args.to_step > len(PIPELINE) or args.from_step > args.to_step:
        raise ValueError(f"Invalid step range: {args.from_step}..{args.to_step}")

    # Run selected range
    for i in range(args.from_step - 1, args.to_step):
        step_name, script = PIPELINE[i]
        run_step(step_name, script, args.python, dry_run=args.dry_run)

    print("\n" + "=" * 80)
    print("🎉 Pipeline finished successfully!")
    print("=" * 80)
    check_expected_files()


if __name__ == "__main__":
    main()
