"""
Visualization: training curves, comparison charts, attention heatmaps,
denoising before/after, parameter/inference comparison, ablation study charts.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image

from model import get_model


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Unified style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (8, 5),
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

COLORS = {
    "DnCNN": "#3498db",
    "U-Net": "#2ecc71",
    "Ours": "#e74c3c",
    "Ablation A (no attn)": "#95a5a6",
    "Ablation B (ch only)": "#f39c12",
    "Ablation C (CBAM)": "#9b59b6",
}


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(results: dict[str, dict], save_path: str | None = None):
    """
    Plot training & validation loss curves for all models.

    Args:
        results: Dict from train_all_models() -> {model_name: {train_losses, val_losses, ...}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, r in results.items():
        color = COLORS.get(name, None)
        axes[0].plot(r["train_losses"], label=name, color=color, linewidth=1.2)
        axes[1].plot(r["val_losses"], label=name, color=color, linewidth=1.2)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Training & Validation Curves", fontsize=14)
    _save(fig, save_path or "training_curves.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. PSNR / SSIM comparison bar charts
# ---------------------------------------------------------------------------

def plot_psnr_comparison(eval_results: dict[str, dict], save_path: str | None = None):
    """Grouped bar chart: PSNR for each method at each sigma."""
    sigmas = [15.0, 25.0, 50.0]
    methods = [m for m in ["DnCNN", "U-Net", "Ours"] if m in eval_results]

    if not methods:
        print("No evaluation results for PSNR comparison.")
        return

    x = np.arange(len(sigmas))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, method in enumerate(methods):
        psnr_vals = [eval_results[method]["sigmas"][s]["psnr"] for s in sigmas]
        bars = ax.bar(x + i * width, psnr_vals, width, label=method,
                      color=COLORS.get(method, None), edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, psnr_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Noise Level σ")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR Comparison Across Noise Levels")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"σ={s:.0f}" for s in sigmas])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    _save(fig, save_path or "psnr_comparison.png")
    plt.close(fig)


def plot_ssim_comparison(eval_results: dict[str, dict], save_path: str | None = None):
    """Grouped bar chart: SSIM for each method at each sigma."""
    sigmas = [15.0, 25.0, 50.0]
    methods = [m for m in ["DnCNN", "U-Net", "Ours"] if m in eval_results]

    if not methods:
        return

    x = np.arange(len(sigmas))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, method in enumerate(methods):
        ssim_vals = [eval_results[method]["sigmas"][s]["ssim"] for s in sigmas]
        bars = ax.bar(x + i * width, ssim_vals, width, label=method,
                      color=COLORS.get(method, None), edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, ssim_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Noise Level σ")
    ax.set_ylabel("SSIM")
    ax.set_title("SSIM Comparison Across Noise Levels")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"σ={s:.0f}" for s in sigmas])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    _save(fig, save_path or "ssim_comparison.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Parameter count & inference time comparison
# ---------------------------------------------------------------------------

def plot_efficiency_comparison(eval_results: dict[str, dict], save_path: str | None = None):
    """Dual bar chart: parameter count and inference time."""
    methods = list(eval_results.keys())
    params_vals = [eval_results[m].get("params", 0) / 1e6 for m in methods]
    time_vals = [eval_results[m].get("inference_time_ms", 0) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    bars1 = ax1.bar(methods, params_vals, color=[COLORS.get(m, "#7f8c8d") for m in methods],
                    edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars1, params_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.2f}M", ha="center", va="bottom", fontsize=9)
    ax1.set_title("Parameter Count")
    ax1.set_ylabel("Millions")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(methods, time_vals, color=[COLORS.get(m, "#7f8c8d") for m in methods],
                    edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars2, time_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}ms", ha="center", va="bottom", fontsize=9)
    ax2.set_title("Inference Time (256×256)")
    ax2.set_ylabel("Milliseconds")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Model Efficiency Comparison", fontsize=14)
    _save(fig, save_path or "efficiency_comparison.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Ablation study chart
# ---------------------------------------------------------------------------

def plot_ablation_study(eval_results: dict[str, dict], sigma: float = 25.0,
                        save_path: str | None = None):
    """
    Ablation study bar chart (PSNR, SSIM, Params) at a specific sigma.
    """
    ablation_keys = [k for k in eval_results if "Ablation" in k or k == "Ours"]
    if not ablation_keys:
        print("No ablation results found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    psnr_vals = []
    ssim_vals = []
    param_vals = []
    labels = []

    for k in ablation_keys:
        r = eval_results[k]
        labels.append(k)
        param_vals.append(r.get("params", 0) / 1e6)
        if sigma in r.get("sigmas", {}):
            psnr_vals.append(r["sigmas"][sigma]["psnr"])
            ssim_vals.append(r["sigmas"][sigma]["ssim"])
        else:
            psnr_vals.append(0)
            ssim_vals.append(0)

    colors = [COLORS.get(l, "#95a5a6") for l in labels]

    # PSNR
    bars = axes[0].bar(labels, psnr_vals, color=colors, edgecolor="white")
    for b, v in zip(bars, psnr_vals):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_title(f"PSNR (σ={sigma:.0f})")
    axes[0].set_ylabel("dB")
    axes[0].tick_params(axis="x", rotation=25)

    # SSIM
    bars = axes[1].bar(labels, ssim_vals, color=colors, edgecolor="white")
    for b, v in zip(bars, ssim_vals):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_title(f"SSIM (σ={sigma:.0f})")
    axes[1].tick_params(axis="x", rotation=25)

    # Params
    bars = axes[2].bar(labels, param_vals, color=colors, edgecolor="white")
    for b, v in zip(bars, param_vals):
        axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                     f"{v:.2f}M", ha="center", va="bottom", fontsize=8)
    axes[2].set_title("Parameter Count")
    axes[2].set_ylabel("Millions")
    axes[2].tick_params(axis="x", rotation=25)

    fig.suptitle("Ablation Study Results", fontsize=14)
    _save(fig, save_path or "ablation_study.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Denoising before/after comparison
# ---------------------------------------------------------------------------

def plot_denoising_examples(
    model: nn.Module,
    test_dir: str,
    sigma: float = 25.0,
    num_examples: int = 4,
    device: str = "cpu",
    save_path: str | None = None,
):
    """
    Show noisy vs denoised vs clean for several test images.
    """
    from dataset import DenoisingDataset

    dataset = DenoisingDataset(
        image_dir=test_dir,
        crop_size=256,
        sigma=[sigma],
        augment=False,
        grayscale=True,
    )

    model.to(device)
    model.eval()

    indices = np.random.choice(min(len(dataset.paths), 20), num_examples, replace=False)

    fig, axes = plt.subplots(num_examples, 3, figsize=(9, 3 * num_examples))

    if num_examples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        img = Image.open(dataset.paths[idx]).convert("L")
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Crop center
        h, w = img_np.shape
        ch, cw = min(h, 256), min(w, 256)
        top, left = (h - ch) // 2, (w - cw) // 2
        clean = img_np[top:top + ch, left:left + cw]

        # Add noise
        noise = np.random.randn(*clean.shape).astype(np.float32) * (sigma / 255.0)
        noisy = np.clip(clean + noise, 0, 1)

        # Denoise
        noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(model, 'attention_type'):
                denoised_t, _ = model(noisy_t)
            else:
                denoised_t = model(noisy_t)
        denoised = denoised_t.squeeze().cpu().numpy()
        denoised = np.clip(denoised, 0, 1)

        # PSNR
        psnr_val = peak_signal_noise_ratio_from_np(clean, denoised)

        axes[row, 0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Noisy (σ={sigma:.0f})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(denoised, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Denoised (PSNR={psnr_val:.1f}dB)")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(clean, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("Clean (Ground Truth)")
        axes[row, 2].axis("off")

    fig.suptitle(f"Denoising Results (σ={sigma:.0f})", fontsize=14)
    _save(fig, save_path or f"denoising_examples_sigma{sigma:.0f}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Attention heatmap visualization
# ---------------------------------------------------------------------------

def plot_attention_heatmaps(
    model: nn.Module,
    test_image_dir: str,
    sigma: float = 25.0,
    device: str = "cpu",
    save_path: str | None = None,
):
    """
    Visualize attention weights from LMAB modules.
    Shows which spatial regions the model focuses on.

    We register hooks on the spatial attention layers of LMAB to extract
    attention maps, then overlay them on the input image.
    """
    attention_maps = []

    def hook_fn(module, input_tensor, output_tensor):
        # output_tensor is the attention-weighted feature map
        # Compute mean attention across channels for visualization
        att_map = output_tensor.detach().abs().mean(dim=1, keepdim=True)
        attention_maps.append(att_map)

    # Register hooks on spatial attention modules
    hooks = []
    for module in model.modules():
        if isinstance(module, SpatialAttention):
            hooks.append(module.register_forward_hook(hook_fn))

    # Load a test image
    from dataset import DenoisingDataset
    dataset = DenoisingDataset(
        image_dir=test_image_dir,
        crop_size=256,
        sigma=[sigma],
        augment=False,
        grayscale=True,
    )
    noisy, clean = dataset[0]
    noisy = noisy.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        _, _ = model(noisy)

    # Remove hooks
    for h in hooks:
        h.remove()

    if not attention_maps:
        print("No attention maps captured. Make sure the model has SpatialAttention modules.")
        return

    # Visualize first few attention maps
    num_blocks = min(len(attention_maps), 8)
    cols = 4
    rows = (num_blocks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.8))
    axes = axes.flatten() if num_blocks > 1 else [axes]

    input_img = noisy.squeeze().cpu().numpy()

    for i in range(num_blocks):
        att = attention_maps[i].squeeze().cpu().numpy()
        # Resize attention map to match input size if needed
        if att.shape != input_img.shape:
            att = np.array(Image.fromarray(att).resize(
                (input_img.shape[1], input_img.shape[0]), Image.BILINEAR
            ))
        # Normalize
        att = (att - att.min()) / (att.max() - att.min() + 1e-8)

        axes[i].imshow(input_img, cmap="gray", vmin=0, vmax=1)
        im = axes[i].imshow(att, cmap="hot", alpha=0.6, vmin=0, vmax=1)
        axes[i].set_title(f"Block {i+1}")
        axes[i].axis("off")

    for i in range(num_blocks, len(axes)):
        axes[i].axis("off")

    fig.colorbar(im, ax=axes[:num_blocks].tolist(), shrink=0.8, label="Attention")
    fig.suptitle(f"LMAB Attention Heatmaps Across Residual Blocks (σ={sigma:.0f})", fontsize=14)
    _save(fig, save_path or f"attention_heatmaps_sigma{sigma:.0f}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Comprehensive results summary (publication-style table)
# ---------------------------------------------------------------------------

def plot_results_table(eval_results: dict[str, dict], save_path: str | None = None):
    """
    Render a publication-style comparison table as an image.
    Table 1: Performance across noise levels.
    """
    sigmas = [15.0, 25.0, 50.0]
    methods = [m for m in ["DnCNN", "U-Net", "Ours"] if m in eval_results]
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")

    col_labels = ["Method", "σ=15", "σ=25", "σ=50", "Params", "Inf Time"]
    cell_text = []
    for m in methods:
        r = eval_results[m]
        row = [m]
        for s in sigmas:
            if s in r.get("sigmas", {}):
                entry = r["sigmas"][s]
                row.append(f"{entry['psnr']:.2f}/{entry['ssim']:.3f}")
            else:
                row.append("-")
        row.append(f"{r.get('params', 0)/1e6:.2f}M")
        row.append(f"{r.get('inference_time_ms', 0):.1f}ms")
        cell_text.append(row)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style first column
    for i in range(len(methods)):
        table[i + 1, 0].set_facecolor("#ecf0f1")

    ax.set_title("Table 1: Denoising Performance Comparison (PSNR/SSIM)", fontsize=13, pad=20)
    _save(fig, save_path or "results_table.png")
    plt.close(fig)


def plot_ablation_table(eval_results: dict[str, dict], sigma: float = 25.0,
                        save_path: str | None = None):
    """Render ablation study table as an image."""
    ablation_keys = [k for k in eval_results if "Ablation" in k or k == "Ours"]
    if not ablation_keys:
        return

    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.axis("off")

    col_labels = ["Configuration", "PSNR (dB)", "SSIM", "Parameters"]
    cell_text = []
    for k in ablation_keys:
        r = eval_results[k]
        psnr = r["sigmas"].get(sigma, {}).get("psnr", "-")
        ssim = r["sigmas"].get(sigma, {}).get("ssim", "-")
        params = f"{r.get('params', 0)/1e6:.2f}M"
        cell_text.append([k, f"{psnr}", f"{ssim}", params])

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(len(ablation_keys)):
        table[i + 1, 0].set_facecolor("#ecf0f1")

    ax.set_title(f"Table 2: Ablation Study Results (σ={sigma:.0f})", fontsize=13, pad=20)
    _save(fig, save_path or "ablation_table.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Generate ALL figures
# ---------------------------------------------------------------------------

def generate_all_figures(
    training_results: dict[str, dict] | None = None,
    eval_results: dict[str, dict] | None = None,
    model: nn.Module | None = None,
    test_dir: str = "data/BSD68",
    device: str = "cpu",
    output_dir: str | None = None,
):
    """
    Generate all charts and visualizations for the report.

    Args:
        training_results: From train_all_models()
        eval_results: From evaluate_all()
        model: Trained 'Ours' model for heatmap & denoising viz
        test_dir: Test dataset directory
        device: torch device
        output_dir: Output directory for figures
    """
    out = output_dir or OUTPUT_DIR
    os.makedirs(out, exist_ok=True)

    generated = []

    # 1. Training curves
    if training_results:
        print("Generating training curves...")
        path = os.path.join(out, "training_curves.png")
        plot_training_curves(training_results, save_path=path)
        generated.append(path)

    # 2-4. Comparison charts (require eval_results)
    if eval_results:
        print("Generating PSNR comparison chart...")
        path = os.path.join(out, "psnr_comparison.png")
        plot_psnr_comparison(eval_results, save_path=path)
        generated.append(path)

        print("Generating SSIM comparison chart...")
        path = os.path.join(out, "ssim_comparison.png")
        plot_ssim_comparison(eval_results, save_path=path)
        generated.append(path)

        print("Generating efficiency comparison chart...")
        path = os.path.join(out, "efficiency_comparison.png")
        plot_efficiency_comparison(eval_results, save_path=path)
        generated.append(path)

        print("Generating ablation study chart...")
        path = os.path.join(out, "ablation_study.png")
        plot_ablation_study(eval_results, save_path=path)
        generated.append(path)

        print("Generating results summary table...")
        path = os.path.join(out, "results_table.png")
        plot_results_table(eval_results, save_path=path)
        generated.append(path)

        print("Generating ablation summary table...")
        path = os.path.join(out, "ablation_table.png")
        plot_ablation_table(eval_results, save_path=path)
        generated.append(path)

    # 5-6. Image-level visualizations (require model + test_dir)
    if model is not None and test_dir and os.path.isdir(test_dir):
        for sigma in [15.0, 25.0, 50.0]:
            print(f"Generating denoising examples (σ={sigma:.0f})...")
            path = os.path.join(out, f"denoising_examples_sigma{sigma:.0f}.png")
            plot_denoising_examples(model, test_dir, sigma=sigma,
                                    device=device, save_path=path)
            generated.append(path)

        print("Generating attention heatmaps...")
        path = os.path.join(out, "attention_heatmaps.png")
        plot_attention_heatmaps(model, test_dir, sigma=25.0,
                                device=device, save_path=path)
        generated.append(path)

    print(f"\nGenerated {len(generated)} figures:")
    for g in generated:
        print(f"  {g}")

    return generated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, path: str):
    """Save figure to the given path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
    print(f"  Saved: {path}")


def peak_signal_noise_ratio_from_np(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Compute PSNR between two numpy arrays in [0, 1]."""
    mse = np.mean((clean - denoised) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


# Import needed for hook registration
from model import SpatialAttention
