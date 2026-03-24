"""
Utility helpers for WorldCache.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def load_npz_cache(npz_path: str) -> dict:
    """
    Load a worldcache.npz file and return its contents as a plain dict.

    Keys present in the archive:
      frame_indices        : int32 array of all frame indices
      anchor_refs          : int32 array mapping each frame to its anchor index
      anchor_frame_indices : int32 array of indices that are anchor frames
      anchor_frames        : uint8 array of shape (N_anchors, H, W, C)
    """
    data = np.load(npz_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def reconstruct_cache_map(npz_data: dict) -> dict:
    """
    Rebuild {frame_idx: anchor_idx} mapping from loaded NPZ data.
    """
    indices = npz_data["frame_indices"]
    refs = npz_data["anchor_refs"]
    return {int(i): int(r) for i, r in zip(indices, refs)}


def build_kv_reuse_schedule(
    cache_map: dict,
) -> List[Tuple[int, Optional[int]]]:
    """
    Build a list of (frame_idx, reuse_from) tuples for world-model KV-cache
    reuse scheduling.

    For anchor frames ``reuse_from`` is None (compute KV from scratch).
    For cached frames ``reuse_from`` is the anchor frame index whose KV cache
    should be reused.
    """
    schedule = []
    for idx in sorted(cache_map):
        anchor = cache_map[idx]
        if anchor == idx:
            schedule.append((idx, None))
        else:
            schedule.append((idx, anchor))
    return schedule


def estimate_ram_savings(
    total_frames: int,
    anchor_frames: int,
    frame_shape: Tuple[int, int, int] = (256, 256, 3),
    dtype_bytes: int = 1,
) -> dict:
    """
    Estimate RAM savings from WorldCache caching.

    Returns a dict with:
      bytes_without_cache, bytes_with_cache, bytes_saved, pct_reduction
    """
    bytes_per_frame = frame_shape[0] * frame_shape[1] * frame_shape[2] * dtype_bytes
    without = total_frames * bytes_per_frame
    with_cache = anchor_frames * bytes_per_frame
    saved = without - with_cache
    pct = (saved / without * 100) if without > 0 else 0.0
    return {
        "bytes_without_cache": without,
        "bytes_with_cache": with_cache,
        "bytes_saved": saved,
        "pct_reduction": pct,
    }


def generate_synthetic_video(
    n_frames: int = 100,
    height: int = 64,
    width: int = 64,
    scene_change_every: int = 30,
    seed: int = 0,
) -> List[np.ndarray]:
    """
    Generate a list of synthetic video frames for testing.

    Creates slow-motion segments (high frame-to-frame similarity) separated
    by hard cuts (scene changes).
    """
    rng = np.random.default_rng(seed)
    frames = []
    base = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)

    for i in range(n_frames):
        if i > 0 and i % scene_change_every == 0:
            base = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)

        noise = rng.integers(-10, 11, (height, width, 3), dtype=np.int16)
        frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)

    return frames


def format_bytes(n: int) -> str:
    """Human-readable byte size string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} PB"
