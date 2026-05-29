#!/usr/bin/env python3
"""
Quick demo: trains models briefly on synthetic data and generates all charts.
Use this to verify the pipeline works end-to-end.
"""

import os, sys, shutil, torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import _create_synthetic_dataset, create_dataloaders, create_test_loader, DATA_ROOT
from train import train
from evaluate import evaluate_model, print_summary_table
from visualize import generate_all_figures
from model import get_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    # Create synthetic data
    print("\n=== Creating synthetic datasets ===")
    train_dir = _create_synthetic_dataset("DemoTrain", 20, size=128)
    test_dir = _create_synthetic_dataset("DemoTest", 8, size=128)

    training_results = {}
    eval_results = {}

    # Train each model for 3 epochs (fast demo)
    configs = [
        ("dncnn", "DnCNN", "dncnn", {}),
        ("unet", "U-Net", "unet", {}),
        ("ours", "Ours", "ours", {"attention_type": "lmab"}),
        ("ours_abl_None", "Ablation A (no attn)", "ours", {"attention_type": None}),
        ("ours_abl_channel_only", "Ablation B (ch only)", "ours", {"attention_type": "channel_only"}),
        ("ours_abl_cbam", "Ablation C (CBAM)", "ours", {"attention_type": "cbam"}),
    ]

    for ckpt_name, display_name, model_name, kwargs in configs:
        print(f"\n--- Training {display_name} ({model_name}) ---")
        result = train(
            model_name=model_name,
            train_dir=train_dir,
            crop_size=40,
            sigma=25.0,
            batch_size=8,
            epochs=3,
            lr=1e-3,
            lr_step=10,
            num_workers=0,
            device=device,
            **kwargs,
        )
        training_results[display_name] = result

        # Rename checkpoint
        ckpt_dir = "checkpoints"
        src = os.path.join(ckpt_dir, f"{model_name}_best.pth")
        dst = os.path.join(ckpt_dir, f"{ckpt_name}.pth")
        if os.path.exists(src):
            shutil.copy2(src, dst)

        # Evaluate
        model = get_model(model_name, **kwargs)
        state = torch.load(dst if os.path.exists(dst) else src, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        eval_result = evaluate_model(
            model, display_name, test_dir,
            sigmas=[15.0, 25.0, 50.0], device=device,
        )
        eval_results[display_name] = eval_result

    print_summary_table(eval_results)

    # Generate figures
    print("\n=== Generating figures ===")
    ours_model = get_model("ours", attention_type="lmab")
    state = torch.load(os.path.join("checkpoints", "ours_best.pth"), map_location=device, weights_only=True)
    ours_model.load_state_dict(state["model_state_dict"])
    ours_model.to(device)

    generated = generate_all_figures(
        training_results=training_results,
        eval_results=eval_results,
        model=ours_model,
        test_dir=test_dir,
        device=device,
    )

    print(f"\nDone! Generated {len(generated)} figures in figures/")
    for g in generated:
        print(f"  {g}")

if __name__ == "__main__":
    main()
