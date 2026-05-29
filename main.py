#!/usr/bin/env python3
"""
CNN + Attention Mechanism for Image Denoising
==============================================
End-to-end pipeline: data download → training → evaluation → visualization.

Usage:
  python main.py                    # Run full pipeline
  python main.py --skip-download    # Skip dataset download
  python main.py --skip-train       # Skip training (eval + viz only)
  python main.py --epochs 30        # Train with 30 epochs
  python main.py --quick            # Quick demo with reduced settings
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import download_bsd68, download_set12, DATA_ROOT
from train import train, train_all_models
from evaluate import evaluate_model, evaluate_all, print_summary_table
from visualize import generate_all_figures
from model import get_model, count_parameters


def parse_args():
    p = argparse.ArgumentParser(description="CNN+Attention Image Denoising")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--crop-size", type=int, default=50, help="Patch crop size")
    p.add_argument("--sigma", type=float, default=25.0, help="Primary noise level for training")
    p.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    p.add_argument("--skip-train", action="store_true", help="Skip training")
    p.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    p.add_argument("--quick", action="store_true", help="Quick demo with reduced settings")
    p.add_argument("--output-dir", type=str, default="figures", help="Output directory for figures")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Quick mode: reduce training for fast demo
    if args.quick:
        args.epochs = 10
        args.batch_size = 8
        args.crop_size = 40
        print("QUICK MODE: reduced epochs/batch/size for fast demo")

    # ------------------------------------------------------------------
    # Step 1: Download datasets
    # ------------------------------------------------------------------
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("Step 1: Downloading datasets")
        print("=" * 60)

        bsd68_dir = download_bsd68()
        print(f"BSD68 directory: {bsd68_dir}")

        # Split BSD68 into train/test (40/28) as per the spec
        all_images = sorted([
            f for f in os.listdir(bsd68_dir)
            if f.lower().endswith((".png", ".jpg", ".bmp"))
        ])
        np.random.seed(42)
        np.random.shuffle(all_images)
        train_imgs = all_images[:40]
        test_imgs = all_images[40:68]

        train_dir = os.path.join(DATA_ROOT, "BSD68_train")
        test_dir = os.path.join(DATA_ROOT, "BSD68_test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        import shutil
        for f in train_imgs:
            src = os.path.join(bsd68_dir, f)
            if not os.path.exists(os.path.join(train_dir, f)):
                shutil.copy2(src, os.path.join(train_dir, f))
        for f in test_imgs:
            src = os.path.join(bsd68_dir, f)
            if not os.path.exists(os.path.join(test_dir, f)):
                shutil.copy2(src, os.path.join(test_dir, f))

        print(f"Training images: {len(os.listdir(train_dir))}")
        print(f"Test images: {len(os.listdir(test_dir))}")

        # Try to get Set12 for extra testing
        try:
            set12_dir = download_set12()
            print(f"Set12 directory: {set12_dir}")
        except Exception:
            set12_dir = test_dir  # fallback to BSD68 test split
            print("Set12 not available, using BSD68 test split.")
    else:
        train_dir = os.path.join(DATA_ROOT, "BSD68_train")
        test_dir = os.path.join(DATA_ROOT, "BSD68_test")
        bsd68_dir = os.path.join(DATA_ROOT, "BSD68")

        # If no train/test split exists, use BSD68 directly
        if not os.path.isdir(train_dir):
            train_dir = bsd68_dir
            test_dir = bsd68_dir

    # ------------------------------------------------------------------
    # Step 2: Train models
    # ------------------------------------------------------------------
    training_results = {}
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("Step 2: Training Models")
        print("=" * 60)

        common = dict(
            train_dir=train_dir,
            crop_size=args.crop_size,
            sigma=args.sigma,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            num_workers=args.num_workers,
            device=device,
        )

        training_results = train_all_models(**common)

        # Train ablation models
        print("\n" + "=" * 60)
        print("Training Ablation Models")
        print("=" * 60)

        ablation_configs = [
            ("ours_abl_None", "Ablation A (no attn)", "ours", {"attention_type": None}),
            ("ours_abl_channel_only", "Ablation B (ch only)", "ours", {"attention_type": "channel_only"}),
            ("ours_abl_cbam", "Ablation C (CBAM)", "ours", {"attention_type": "cbam"}),
        ]

        for ckpt_name, display_name, model_name, model_kwargs in ablation_configs:
            print(f"\n--- Training {display_name} ---")
            result = train(
                model_name=model_name,
                train_dir=train_dir,
                crop_size=args.crop_size,
                sigma=args.sigma,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                num_workers=args.num_workers,
                device=device,
                **model_kwargs,
            )
            training_results[display_name] = result
            # Rename checkpoint for ablation
            import shutil
            src = os.path.join("checkpoints", "ours_best.pth")
            dst = os.path.join("checkpoints", f"{ckpt_name}.pth")
            if os.path.exists(src):
                shutil.copy2(src, dst)

    # ------------------------------------------------------------------
    # Step 3: Evaluate models
    # ------------------------------------------------------------------
    eval_results = {}
    if not args.skip_eval:
        print("\n" + "=" * 60)
        print("Step 3: Evaluation")
        print("=" * 60)

        eval_results = evaluate_all(
            checkpoint_dir="checkpoints",
            test_dir=test_dir,
            sigmas=[15.0, 25.0, 50.0],
            device=device,
        )
        print_summary_table(eval_results)

    # ------------------------------------------------------------------
    # Step 4: Generate figures
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Generating Figures")
    print("=" * 60)

    # Load the best 'Ours' model for heatmap & denoising examples
    ours_model = None
    ckpt_path = os.path.join("checkpoints", "ours_best.pth")
    if os.path.exists(ckpt_path):
        ours_model = get_model("ours", attention_type="lmab")
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        ours_model.load_state_dict(state["model_state_dict"])
        ours_model.to(device)

    generated = generate_all_figures(
        training_results=training_results,
        eval_results=eval_results,
        model=ours_model,
        test_dir=test_dir,
        device=device,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Figures saved to: {os.path.abspath(args.output_dir)}/")
    print(f"Checkpoints saved to: {os.path.abspath('checkpoints')}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
