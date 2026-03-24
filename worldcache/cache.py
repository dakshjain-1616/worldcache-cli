"""
WorldCache frame cache.

Stores anchor frames and maps redundant frames to the closest cached anchor.
RAM savings come from only materialising anchor frames in memory; all other
frames are represented by a (anchor_idx, similarity_score) tuple.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .hasher import TemporalHasher, HashAlgorithm

# Similarity threshold: frames with similarity >= this value are considered
# "cache-hits" and won't be stored as separate anchor frames.
DEFAULT_SIMILARITY_THRESHOLD = float(
    os.getenv("WORLDCACHE_SIMILARITY_THRESHOLD", "0.92")
)

# Maximum number of anchor frames to keep in the sliding comparison window.
DEFAULT_WINDOW_SIZE = int(os.getenv("WORLDCACHE_WINDOW_SIZE", "32"))


@dataclass
class CacheEntry:
    """Represents a single frame's cache status."""

    frame_idx: int
    is_anchor: bool
    anchor_idx: int  # equals frame_idx when is_anchor=True
    similarity: float  # 1.0 for anchor frames
    frame_hash: int
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)

    @property
    def is_cached(self) -> bool:
        return not self.is_anchor


@dataclass
class CacheStats:
    """Aggregate statistics for a cache run."""

    total_frames: int = 0
    anchor_frames: int = 0
    cached_frames: int = 0
    total_pixels_saved: int = 0
    frame_width: int = 0
    frame_height: int = 0
    frame_channels: int = 3

    @property
    def cache_hit_rate(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.cached_frames / self.total_frames

    @property
    def ram_reduction_pct(self) -> float:
        return self.cache_hit_rate * 100.0

    @property
    def bytes_saved(self) -> int:
        if self.frame_width == 0 or self.frame_height == 0:
            return 0
        bytes_per_frame = (
            self.frame_width * self.frame_height * self.frame_channels
        )
        return self.cached_frames * bytes_per_frame

    @property
    def bytes_used(self) -> int:
        if self.frame_width == 0 or self.frame_height == 0:
            return 0
        bytes_per_frame = (
            self.frame_width * self.frame_height * self.frame_channels
        )
        return self.anchor_frames * bytes_per_frame

    def summary(self) -> str:
        lines = [
            "WorldCache Statistics",
            "─" * 40,
            f"Total frames    : {self.total_frames:,}",
            f"Anchor frames   : {self.anchor_frames:,}",
            f"Cached frames   : {self.cached_frames:,}",
            f"Cache hit rate  : {self.cache_hit_rate:.1%}",
            f"RAM reduction   : {self.ram_reduction_pct:.1f}%",
        ]
        if self.bytes_saved > 0:
            mb_saved = self.bytes_saved / (1024**2)
            mb_used = self.bytes_used / (1024**2)
            lines.append(f"Bytes saved     : {mb_saved:.1f} MB")
            lines.append(f"Bytes used      : {mb_used:.1f} MB")
        return "\n".join(lines)


class FrameCache:
    """
    Content-aware frame cache using temporal similarity hashing.

    Algorithm (WorldCache-style):
      1. Compute perceptual hash for incoming frame.
      2. Compare against hashes in the sliding window of recent anchor frames.
      3. If best match similarity >= threshold → cache hit (reference to anchor).
      4. Otherwise → new anchor frame stored in window.

    The result is a dense lookup table: frame_idx → CacheEntry, which downstream
    world-model training code can use to reuse KV-cache entries across similar frames.
    """

    def __init__(
        self,
        algorithm: HashAlgorithm = HashAlgorithm.DHASH,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        window_size: int = DEFAULT_WINDOW_SIZE,
        hash_size: int = 16,
    ) -> None:
        self.hasher = TemporalHasher(algorithm=algorithm, hash_size=hash_size)
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

        # Sliding window: list of (frame_idx, hash_int) for recent anchors
        self._window: List[Tuple[int, int]] = []
        # Full lookup table
        self._entries: Dict[int, CacheEntry] = {}
        # Materialised anchor frames (only anchors are stored)
        self._anchor_frames: Dict[int, np.ndarray] = {}
        self._stats = CacheStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        store_frame: bool = True,
    ) -> CacheEntry:
        """
        Process one video frame and return its cache entry.

        Args:
            frame: H×W×C uint8 numpy array.
            frame_idx: Zero-based frame index in the video.
            store_frame: Whether to materialise anchor frames in RAM.

        Returns:
            CacheEntry describing whether this frame is an anchor or a cache hit.
        """
        h, w = frame.shape[:2]
        c = frame.shape[2] if frame.ndim == 3 else 1

        if self._stats.total_frames == 0:
            self._stats.frame_width = w
            self._stats.frame_height = h
            self._stats.frame_channels = c

        frame_hash = self.hasher.hash(frame)
        best_anchor_idx, best_sim = self._find_best_match(frame_hash)

        is_anchor = best_sim < self.similarity_threshold

        if is_anchor:
            anchor_idx = frame_idx
            similarity = 1.0
            # Add to sliding window
            self._window.append((frame_idx, frame_hash))
            if len(self._window) > self.window_size:
                self._window.pop(0)
            if store_frame:
                self._anchor_frames[frame_idx] = frame.copy()
            self._stats.anchor_frames += 1
        else:
            anchor_idx = best_anchor_idx
            similarity = best_sim
            self._stats.cached_frames += 1
            self._stats.total_pixels_saved += w * h * c

        entry = CacheEntry(
            frame_idx=frame_idx,
            is_anchor=is_anchor,
            anchor_idx=anchor_idx,
            similarity=similarity,
            frame_hash=frame_hash,
        )
        self._entries[frame_idx] = entry
        self._stats.total_frames += 1
        return entry

    def get_entry(self, frame_idx: int) -> Optional[CacheEntry]:
        return self._entries.get(frame_idx)

    def get_anchor_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return the materialised numpy array for an anchor frame (or None)."""
        return self._anchor_frames.get(frame_idx)

    def resolve_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Resolve any frame index to an array by following the cache pointer.
        Returns anchor frame data for both anchor and cached frames.
        """
        entry = self._entries.get(frame_idx)
        if entry is None:
            return None
        return self._anchor_frames.get(entry.anchor_idx)

    @property
    def stats(self) -> CacheStats:
        return self._stats

    @property
    def entries(self) -> Dict[int, CacheEntry]:
        return dict(self._entries)

    def cache_map(self) -> Dict[int, int]:
        """Return mapping: frame_idx → anchor_idx for all frames."""
        return {idx: e.anchor_idx for idx, e in self._entries.items()}

    def anchor_indices(self) -> List[int]:
        """Return sorted list of all anchor frame indices."""
        return sorted(idx for idx, e in self._entries.items() if e.is_anchor)

    def reset(self) -> None:
        self._window.clear()
        self._entries.clear()
        self._anchor_frames.clear()
        self._stats = CacheStats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_best_match(self, frame_hash: int) -> Tuple[int, float]:
        """
        Search the sliding window for the most similar anchor frame.
        Returns (anchor_idx, similarity_score).
        """
        if not self._window:
            return -1, 0.0

        best_idx = -1
        best_sim = 0.0
        for anchor_idx, anchor_hash in self._window:
            sim = self.hasher.similarity(frame_hash, anchor_hash)
            if sim > best_sim:
                best_sim = sim
                best_idx = anchor_idx

        return best_idx, best_sim
