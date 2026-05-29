"""
Dataset handling: BSD68 download, noise synthesis, patch extraction.

Uses BSD68 (68 grayscale images) split into train/val/test.
Adds Gaussian noise at sigma = 15, 25, 50.
"""

import os
import random
import zipfile
import requests
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

BSD68_URLS = [
    "https://github.com/clausmichele/BSD68-dataset/archive/refs/heads/master.zip",
]

SET12_URLS = [
    "https://github.com/cszn/DnCNN/raw/master/testsets/Set12.zip",
]

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _download_file(url: str, dest: str) -> bool:
    """Download a file with progress indication. Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def _extract_zip(zip_path: str, extract_dir: str) -> bool:
    """Extract a zip file."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        return True
    except Exception as e:
        print(f"  Extract failed: {e}")
        return False


def download_bsd68() -> str:
    """
    Download and extract BSD68 dataset.
    Returns the path to the directory containing the 68 .png images.
    """
    bsd_dir = os.path.join(DATA_ROOT, "BSD68")
    os.makedirs(DATA_ROOT, exist_ok=True)

    # Check if already downloaded
    if os.path.isdir(bsd_dir):
        pngs = [f for f in os.listdir(bsd_dir) if f.endswith((".png", ".jpg", ".bmp"))]
        if len(pngs) >= 68:
            print(f"BSD68 already downloaded: {len(pngs)} images in {bsd_dir}")
            return bsd_dir

    zip_path = os.path.join(DATA_ROOT, "bsd68.zip")
    for url in BSD68_URLS:
        print(f"Downloading BSD68 from {url} ...")
        if _download_file(url, zip_path):
            if _extract_zip(zip_path, DATA_ROOT):
                break
    else:
        # If downloads fail, create a synthetic dataset for demonstration
        print("Automatic download failed. Creating synthetic demo dataset...")
        return _create_synthetic_dataset("BSD68", 68)

    # Find the extracted image directory
    for root, dirs, files in os.walk(DATA_ROOT):
        pngs = [f for f in files if f.endswith((".png", ".jpg", ".bmp"))]
        if len(pngs) >= 68:
            # Move/copy to expected location
            src_dir = root
            if src_dir != bsd_dir:
                import shutil
                if os.path.exists(bsd_dir):
                    shutil.rmtree(bsd_dir)
                os.rename(src_dir, bsd_dir)
            print(f"BSD68 ready: {len(pngs)} images in {bsd_dir}")
            return bsd_dir

    return _create_synthetic_dataset("BSD68", 68)


def download_set12() -> str:
    """Download Set12 test dataset."""
    set12_dir = os.path.join(DATA_ROOT, "Set12")
    os.makedirs(DATA_ROOT, exist_ok=True)

    if os.path.isdir(set12_dir):
        imgs = [f for f in os.listdir(set12_dir) if f.endswith((".png", ".jpg", ".bmp"))]
        if len(imgs) >= 12:
            return set12_dir

    # Try download
    zip_path = os.path.join(DATA_ROOT, "set12.zip")
    for url in SET12_URLS:
        print(f"Downloading Set12 from {url} ...")
        if _download_file(url, zip_path):
            if _extract_zip(zip_path, DATA_ROOT):
                break

    for root, dirs, files in os.walk(DATA_ROOT):
        imgs = [f for f in files if f.endswith((".png", ".jpg", ".bmp"))]
        if len(imgs) >= 12 and "Set12" in root:
            return root

    return _create_synthetic_dataset("Set12", 12)


def _create_synthetic_dataset(name: str, count: int, size: int = 256) -> str:
    """
    Create a synthetic dataset of random images for demonstration.
    Used as fallback when real dataset download fails.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(DATA_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(42)
    for i in range(count):
        # Generate a structured "natural-looking" image using Perlin-noise-like patterns
        img = _random_texture(size)
        path = os.path.join(out_dir, f"{name.lower()}_{i:03d}.png")
        Image.fromarray((img * 255).astype(np.uint8)).save(path)

    print(f"Created synthetic {name} dataset: {count} images in {out_dir}")
    return out_dir


def _random_texture(size: int = 256) -> np.ndarray:
    """Generate a random texture image with varying frequency content."""
    x = np.linspace(0, 4 * np.pi, size)
    y = np.linspace(0, 4 * np.pi, size)
    X, Y = np.meshgrid(x, y)

    img = np.zeros((size, size))
    # Add multiple frequency components for realistic texture
    for freq, amp in [(0.5, 1.0), (1.0, 0.7), (2.0, 0.4), (4.0, 0.2), (8.0, 0.1)]:
        phase_x = np.random.uniform(0, 2 * np.pi)
        phase_y = np.random.uniform(0, 2 * np.pi)
        img += amp * np.sin(freq * X + phase_x) * np.cos(freq * Y + phase_y)

    # Add edges (step functions)
    threshold = np.random.uniform(-0.3, 0.3)
    if random.random() < 0.3:
        img += 0.5 * (np.abs(img) > threshold).astype(np.float64)

    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.astype(np.float64)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class DenoisingDataset(Dataset):
    """
    Dataset for image denoising.

    Args:
        image_dir: Directory containing clean images.
        crop_size: Size of random patches (None = full image).
        sigma: Gaussian noise standard deviation (or list for random choice).
        augment: Apply random flips/rotations.
        grayscale: Convert to grayscale.
    """

    def __init__(
        self,
        image_dir: str,
        crop_size: int | None = 50,
        sigma: float | list[float] = 25.0,
        augment: bool = True,
        grayscale: bool = True,
    ):
        self.crop_size = crop_size
        self.sigma = sigma if isinstance(sigma, list) else [sigma]
        self.augment = augment
        self.grayscale = grayscale

        self.paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ])
        if not self.paths:
            raise RuntimeError(f"No images found in {image_dir}")

        print(f"  Loaded {len(self.paths)} images from {image_dir}")

    def __len__(self) -> int:
        # Return a fixed number of patches per epoch
        return max(len(self.paths) * 10, 400)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Pick a random image
        img_path = random.choice(self.paths)
        img = Image.open(img_path)

        if self.grayscale and img.mode != "L":
            img = img.convert("L")

        img_np = np.array(img, dtype=np.float32) / 255.0

        # Random crop
        if self.crop_size is not None:
            h, w = img_np.shape[:2]
            if h < self.crop_size or w < self.crop_size:
                img_np = np.pad(img_np, (
                    (max(0, self.crop_size - h) // 2, max(0, self.crop_size - h) // 2),
                    (max(0, self.crop_size - w) // 2, max(0, self.crop_size - w) // 2),
                ), mode="reflect")
                h, w = img_np.shape[:2]
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            img_np = img_np[top:top + self.crop_size, left:left + self.crop_size]

        # Ensure shape [H, W] or [C, H, W]
        if img_np.ndim == 2:
            img_np = img_np[np.newaxis, ...]  # [1, H, W]
        else:
            img_np = img_np.transpose(2, 0, 1)  # [C, H, W]

        clean = torch.from_numpy(img_np.copy()).float()

        # Data augmentation
        if self.augment:
            if random.random() < 0.5:
                clean = torch.flip(clean, dims=[-1])  # horizontal flip
            if random.random() < 0.5:
                clean = torch.flip(clean, dims=[-2])  # vertical flip
            if random.random() < 0.5:
                clean = torch.rot90(clean, k=random.randint(0, 3), dims=[-2, -1])

        # Add Gaussian noise
        sigma = random.choice(self.sigma) / 255.0  # convert to [0,1] scale
        noise = torch.randn_like(clean) * sigma
        noisy = clean + noise

        return noisy, clean


def create_dataloaders(
    train_dir: str,
    val_dir: str | None = None,
    crop_size: int = 50,
    sigma: float | list[float] = 25.0,
    batch_size: int = 16,
    num_workers: int = 2,
    val_split: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    from torch.utils.data import random_split

    full_dataset = DenoisingDataset(
        image_dir=train_dir,
        crop_size=crop_size,
        sigma=sigma,
        augment=True,
    )

    if val_dir is not None:
        train_dataset = full_dataset
        val_dataset = DenoisingDataset(
            image_dir=val_dir,
            crop_size=crop_size,
            sigma=sigma,
            augment=False,
        )
    else:
        n_val = max(int(len(full_dataset) * val_split), 1)
        n_train = len(full_dataset) - n_val
        train_dataset, val_dataset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader


def create_test_loader(
    test_dir: str,
    sigma: float = 25.0,
    batch_size: int = 1,
    num_workers: int = 2,
) -> DataLoader:
    """Create test dataloader (full images, no cropping)."""
    dataset = DenoisingDataset(
        image_dir=test_dir,
        crop_size=None,   # full image
        sigma=[sigma],
        augment=False,
    )
    # For testing, use full images without random patch selection
    # Override to get one sample per image
    dataset.__len__ = lambda self=dataset: len(dataset.paths)
    dataset.__getitem__ = _make_test_getitem(dataset)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


def _make_test_getitem(dataset: DenoisingDataset):
    """Factory for test-mode __getitem__ that iterates images sequentially."""
    paths = dataset.paths
    sigma = dataset.sigma
    grayscale = dataset.grayscale

    def __getitem__(idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(paths[idx])
        if grayscale and img.mode != "L":
            img = img.convert("L")
        img_np = np.array(img, dtype=np.float32) / 255.0
        if img_np.ndim == 2:
            img_np = img_np[np.newaxis, ...]
        else:
            img_np = img_np.transpose(2, 0, 1)

        clean = torch.from_numpy(img_np.copy()).float()
        s = sigma[0] / 255.0
        noise = torch.randn_like(clean) * s
        noisy = clean + noise
        return noisy, clean

    return __getitem__
