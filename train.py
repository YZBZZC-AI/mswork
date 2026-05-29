"""
Training pipeline for image denoising models.

Implements:
  - Training loop with configurable parameters
  - Learning rate scheduling (decay by 0.5 every 20 epochs)
  - Early stopping
  - Checkpoint saving
"""

import os
import sys
import time
import logging
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np

from model import get_model, count_parameters
from dataset import create_dataloaders


def setup_logging(log_dir: str, model_name: str) -> logging.Logger:
    """Configure logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.log")

    logger = logging.getLogger(f"train_{model_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging to {log_file}")
    return logger


def train(
    model_name: str = "ours",
    train_dir: str = "data/BSD68",
    val_dir: str | None = None,
    crop_size: int = 50,
    sigma: float = 25.0,
    batch_size: int = 16,
    epochs: int = 50,
    lr: float = 1e-3,
    lr_step: int = 20,
    lr_gamma: float = 0.5,
    early_stop_patience: int = 15,
    num_workers: int = 2,
    device: str | None = None,
    save_dir: str = "checkpoints",
    **model_kwargs,
) -> dict:
    """
    Train a denoising model.

    Returns:
        dict with keys: train_losses, val_losses, best_epoch, best_loss, model_path, training_time
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(save_dir, exist_ok=True)

    log = setup_logging(os.path.join(save_dir, "logs"), model_name)

    log.info(f"\n{'='*60}")
    log.info(f"Training {model_name.upper()} on {device}")
    log.info(f"{'='*60}")

    # Data
    train_loader, val_loader = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        crop_size=crop_size,
        sigma=sigma,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    log.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = get_model(model_name, **model_kwargs).to(device)
    n_params = count_parameters(model)
    log.info(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    criterion = nn.MSELoss()

    # Training state
    train_losses = []
    val_losses = []
    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs} [train]")
        for noisy, clean in pbar:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()

            if model_name in ("ours",):
                output, _ = model(noisy)
            else:
                output = model(noisy)

            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in tqdm(val_loader, desc=f"Epoch {epoch:3d}/{epochs} [val  ]", leave=False):
                noisy, clean = noisy.to(device), clean.to(device)
                if model_name in ("ours",):
                    output, _ = model(noisy)
                else:
                    output = model(noisy)
                val_loss += criterion(output, clean).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        log.info(f"Epoch {epoch:3d} | train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | lr={current_lr:.2e}")

        # Checkpointing
        is_best = avg_val_loss < best_loss
        if is_best:
            best_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_loss,
            }, os.path.join(save_dir, f"{model_name}_best.pth"))
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop_patience:
            log.info(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break

    training_time = time.time() - t_start
    log.info(f"\nTraining completed in {training_time:.1f}s ({training_time/60:.1f} min)")
    log.info(f"Best val_loss: {best_loss:.6f} at epoch {best_epoch}")

    result = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "model_path": os.path.join(save_dir, f"{model_name}_best.pth"),
        "training_time": training_time,
        "parameters": n_params,
    }

    # Save training history
    np.savez(
        os.path.join(save_dir, f"{model_name}_history.npz"),
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        best_epoch=best_epoch,
        best_loss=best_loss,
    )

    return result


def train_all_models(
    train_dir: str = "data/BSD68",
    val_dir: str | None = None,
    device: str | None = None,
    save_dir: str = "checkpoints",
    **kwargs,
) -> dict[str, dict]:
    """
    Train all models for the comparison experiment:
      - DnCNN (baseline)
      - UNet (baseline)
      - Ours (DnCNN + LMAB)
    """
    configs = {
        "dncnn": {"model_name": "dncnn"},
        "unet": {"model_name": "unet"},
        "ours": {"model_name": "ours", "attention_type": "lmab"},
    }

    results = {}
    for name, cfg in configs.items():
        print(f"\n{'#'*60}")
        print(f"# Training {name}")
        print(f"{'#'*60}")
        result = train(
            train_dir=train_dir,
            val_dir=val_dir,
            device=device,
            save_dir=save_dir,
            **cfg,
            **kwargs,
        )
        results[name] = result

    return results
