"""Convenience wrapper to train all supported models with the same settings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = ["baseline", "swin", "foundation"]


def _str2bool(value):
    """Parse flexible boolean command-line values."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def main():
    """Train every supported model one after another."""
    parser = argparse.ArgumentParser(description="Train all supported models.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    parser.add_argument("--technique", choices=["none", "technique1", "technique2", "technique3"], default="none")
    parser.add_argument("--data_root", default="data/isic2019")
    parser.add_argument("--splits_dir", default="data/isic2019/splits/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="outputs/")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", type=_str2bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    for model_name in args.models:
        cmd = [
            sys.executable,
            str(ROOT / "train.py"),
            "--model",
            model_name,
            "--technique",
            args.technique,
            "--data_root",
            args.data_root,
            "--splits_dir",
            args.splits_dir,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--device",
            args.device,
            "--out_dir",
            args.out_dir,
            "--save_every",
            str(args.save_every),
            "--seed",
            str(args.seed),
            "--pretrained",
            str(args.pretrained),
            "--num_workers",
            str(args.num_workers),
        ]
        if args.resume is not None:
            cmd.extend(["--resume", args.resume])

        print(f"=== Training {model_name} ===")
        completed = subprocess.run(cmd, cwd=ROOT)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
