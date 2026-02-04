"""Small helpers for saving example outputs without extra deps."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError("Unsupported image shape for RGB conversion.")


def _normalize_to_uint8(image: np.ndarray, stretch: bool) -> np.ndarray:
    img = image.astype(np.float32)
    if img.size == 0:
        return img.astype(np.uint8)
    min_val = float(img.min())
    max_val = float(img.max())
    if stretch and max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    elif min_val < 0.0 or max_val > 1.0:
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


def save_ppm(image: np.ndarray, path: Path, stretch: bool = False) -> None:
    """Save an image to binary PPM (P6) format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _ensure_rgb(image)
    data = _normalize_to_uint8(rgb, stretch=stretch)
    height, width, _ = data.shape
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(data.tobytes())


def save_volume_slice(
    volume: np.ndarray,
    path: Path,
    axis: int = 0,
    index: Optional[int] = None,
    stretch: bool = False,
) -> None:
    """Save a middle slice from a 3D/4D volume."""
    if volume.ndim == 3:
        data = volume
    elif volume.ndim == 4:
        data = volume
    else:
        raise ValueError("Volume must be 3D or 4D.")
    
    if index is None:
        index = data.shape[axis] // 2
    if axis == 0:
        slice_img = data[index]
    elif axis == 1:
        slice_img = data[:, index, :]
    else:
        slice_img = data[:, :, index]
    
    save_ppm(slice_img, path, stretch=stretch)


def save_ppm_sequence(
    frames: Iterable[np.ndarray],
    directory: Path,
    prefix: str = "frame",
    stretch: bool = False,
) -> None:
    """Save a sequence of frames as numbered PPM files."""
    directory.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        save_ppm(frame, directory / f"{prefix}_{i:03d}.ppm", stretch=stretch)


def save_mp4(
    frames: Iterable[np.ndarray],
    path: Path,
    fps: int = 24,
    stretch: bool = False,
    macro_block_size: int = 1,
) -> None:
    """Save frames to MP4 if imageio is available."""
    try:
        import imageio
        use_v3 = hasattr(imageio, "v3")
    except Exception:
        print("imageio is not available; skipping mp4 export.")
        return
    
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = []
    for frame in frames:
        rgb = _ensure_rgb(frame)
        data = _normalize_to_uint8(rgb, stretch=stretch)
        prepared.append(data)
    
    if use_v3:
        imageio.v3.imwrite(path, prepared, fps=fps, macro_block_size=macro_block_size)
    else:
        imageio.mimsave(path, prepared, fps=fps, macro_block_size=macro_block_size)


"""Small helpers for saving example outputs without extra deps."""
