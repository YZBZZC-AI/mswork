"""
Evaluation: PSNR, SSIM, parameter count, inference time.

Evaluates all models across noise levels sigma = 15, 25, 50.
"""

import os
import time
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model import get_model, count_parameters
from dataset import create_test_loader


def compute_psnr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Compute PSNR between two images in [0, 1] range."""
    data_range = clean.max() - clean.min()
    if data_range < 1e-8:
        data_range = 1.0
    return peak_signal_noise_ratio(clean, denoised, data_range=float(data_range))


def compute_ssim(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Compute SSIM between two grayscale images in [0, 1] range."""
    data_range = clean.max() - clean.min()
    if data_range < 1e-8:
        data_range = 1.0
    return structural_similarity(clean, denoised, data_range=float(data_range))


def measure_inference_time(model: torch.nn.Module, input_tensor: torch.Tensor,
                           device: str, warmup: int = 10, repeats: int = 50) -> float:
    """Measure average inference time in milliseconds."""
    model.eval()
    x = input_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    # Timed runs
    if device == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    return (elapsed / repeats) * 1000  # ms


def evaluate_model(
    model: torch.nn.Module,
    model_name: str,
    test_dir: str,
    sigmas: list[float] = [15.0, 25.0, 50.0],
    device: str = "cpu",
    input_size: tuple = (1, 1, 256, 256),
) -> dict:
    """
    Evaluate a model across multiple noise levels.

    Returns:
        dict with PSNR, SSIM, params, inference_time per sigma level.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")

    model.to(device)
    model.eval()

    results = {
        "model": model_name,
        "params": count_parameters(model),
        "sigmas": {},
    }

    dummy_input = torch.randn(*input_size)
    results["inference_time_ms"] = measure_inference_time(model, dummy_input, device)

    for sigma in sigmas:
        print(f"\n  Sigma = {sigma:.0f}")
        test_loader = create_test_loader(test_dir, sigma=sigma, batch_size=1, num_workers=0)

        psnr_list, ssim_list = [], []
        for noisy, clean in tqdm(test_loader, desc=f"  Testing σ={sigma:.0f}"):
            noisy = noisy.to(device)
            with torch.no_grad():
                if isinstance(model, torch.nn.Module) and hasattr(model, 'attention_type'):
                    output, _ = model(noisy)
                else:
                    output = model(noisy)

            # Move to CPU numpy
            clean_np = clean.squeeze().cpu().numpy()
            denoised_np = output.squeeze().cpu().numpy()
            # Clip to valid range
            denoised_np = np.clip(denoised_np, 0, 1)

            psnr_list.append(compute_psnr(clean_np, denoised_np))
            ssim_list.append(compute_ssim(clean_np, denoised_np))

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        results["sigmas"][sigma] = {
            "psnr": round(avg_psnr, 2),
            "ssim": round(avg_ssim, 3),
            "psnr_std": round(np.std(psnr_list), 2),
            "ssim_std": round(np.std(ssim_list), 3),
        }
        print(f"    PSNR: {avg_psnr:.2f} ± {np.std(psnr_list):.2f} dB")
        print(f"    SSIM: {avg_ssim:.3f} ± {np.std(ssim_list):.3f}")

    return results


def evaluate_all(
    checkpoint_dir: str = "checkpoints",
    test_dir: str = "data/BSD68",
    sigmas: list[float] | None = None,
    device: str | None = None,
) -> dict[str, dict]:
    """
    Evaluate all trained models (DnCNN, UNet, Ours).
    Also evaluates ablation variants if checkpoints exist.
    """
    if sigmas is None:
        sigmas = [15.0, 25.0, 50.0]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}

    # Models to evaluate
    model_configs = {
        "DnCNN": {"name": "dncnn"},
        "U-Net": {"name": "unet"},
        "Ours": {"name": "ours", "attention_type": "lmab"},
    }

    for display_name, cfg in model_configs.items():
        ckpt_path = os.path.join(checkpoint_dir, f"{cfg['name']}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}, skipping {display_name}")
            continue

        model = get_model(**cfg)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])

        result = evaluate_model(
            model, display_name, test_dir,
            sigmas=sigmas, device=device,
        )
        results[display_name] = result

    # Evaluate ablation variants (ours variants with different attention)
    ablation_configs = {
        "Ablation A (no attn)": {"name": "ours", "attention_type": None},
        "Ablation B (ch only)": {"name": "ours", "attention_type": "channel_only"},
        "Ablation C (CBAM)": {"name": "ours", "attention_type": "cbam"},
    }

    for display_name, cfg in ablation_configs.items():
        ckpt_name = f"ours_abl_{cfg['attention_type']}.pth"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}, skipping {display_name}")
            continue

        model = get_model(**cfg)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])

        result = evaluate_model(
            model, display_name, test_dir,
            sigmas=sigmas, device=device,
        )
        results[display_name] = result

    return results


def print_summary_table(results: dict[str, dict]):
    """Print a formatted summary table."""
    if not results:
        print("No results to print.")
        return

    sigmas = [15.0, 25.0, 50.0]
    header = f"{'Method':<25} | " + " | ".join(
        f"σ={s:.0f} (PSNR/SSIM)" for s in sigmas
    ) + " | Params  | Inf Time"
    print(f"\n{header}")
    print("-" * len(header))

    for name, r in results.items():
        row = f"{name:<25} | "
        for s in sigmas:
            if s in r.get("sigmas", {}):
                entry = r["sigmas"][s]
                row += f"{entry['psnr']:.2f}/{entry['ssim']:.3f}  | "
            else:
                row += "  - /  -    | "
        params_k = r.get("params", 0) / 1e6
        inf_time = r.get("inference_time_ms", 0)
        row += f"{params_k:.2f}M   | {inf_time:.1f}ms"
        print(row)
