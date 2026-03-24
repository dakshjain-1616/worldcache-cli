"""
Video processing pipeline for WorldCache.

Reads video frames (via OpenCV or a synthetic source), runs them through
the FrameCache, and serialises results to disk in multiple formats:

  - NPZ  : compressed numpy archive with cache map + anchor frames
  - JSON : human-readable cache map and statistics
  - CSV  : per-frame breakdown for analysis
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

from .cache import CacheStats, FrameCache
from .hasher import HashAlgorithm

# ── Environment knobs ─────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = os.getenv("WORLDCACHE_OUTPUT_DIR", "worldcache_output")
DEFAULT_ALGORITHM = os.getenv("WORLDCACHE_ALGORITHM", "dhash")
DEFAULT_SIMILARITY = float(os.getenv("WORLDCACHE_SIMILARITY_THRESHOLD", "0.92"))
DEFAULT_WINDOW = int(os.getenv("WORLDCACHE_WINDOW_SIZE", "32"))
DEFAULT_HASH_SIZE = int(os.getenv("WORLDCACHE_HASH_SIZE", "16"))
MAX_FRAMES = int(os.getenv("WORLDCACHE_MAX_FRAMES", "0"))  # 0 = unlimited


@dataclass
class ProcessingResult:
    """Returned by VideoProcessor.process()."""

    source: str
    total_frames: int
    anchor_frames: int
    cached_frames: int
    cache_hit_rate: float
    ram_reduction_pct: float
    bytes_saved: int
    elapsed_seconds: float
    output_dir: str
    output_files: List[str]
    stats: CacheStats

    def to_dict(self) -> dict:
        d = {
            "source": self.source,
            "total_frames": self.total_frames,
            "anchor_frames": self.anchor_frames,
            "cached_frames": self.cached_frames,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "ram_reduction_pct": round(self.ram_reduction_pct, 2),
            "bytes_saved": self.bytes_saved,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "output_dir": self.output_dir,
            "output_files": self.output_files,
        }
        return d


class VideoProcessor:
    """
    End-to-end WorldCache processor.

    Usage::

        proc = VideoProcessor(output_dir="out/")
        result = proc.process("my_video.mp4")
        print(result.stats.summary())
    """

    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        algorithm: str = DEFAULT_ALGORITHM,
        similarity_threshold: float = DEFAULT_SIMILARITY,
        window_size: int = DEFAULT_WINDOW,
        hash_size: int = DEFAULT_HASH_SIZE,
        max_frames: int = MAX_FRAMES,
        store_frames: bool = True,
        scene_change_every: int = 40,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.algorithm = HashAlgorithm(algorithm)
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.hash_size = hash_size
        self.max_frames = max_frames
        self.store_frames = store_frames
        self.scene_change_every = scene_change_every
        self.progress_callback = progress_callback

        self._cache = FrameCache(
            algorithm=self.algorithm,
            similarity_threshold=self.similarity_threshold,
            window_size=self.window_size,
            hash_size=self.hash_size,
        )

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def process(self, source: str) -> ProcessingResult:
        """
        Process a video file or directory of frames and write outputs.

        ``source`` may be:
          - A path to a video file (e.g., ``video.mp4``)
          - A path to a directory of image frames (sorted by name)
          - The string ``"synthetic"`` to generate test frames in-memory
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cache.reset()

        t0 = time.time()
        output_files: List[str] = []

        if source == "synthetic":
            frames_iter = self._synthetic_frames(scene_change_every=self.scene_change_every)
        elif Path(source).is_dir():
            frames_iter = self._frames_from_dir(source)
        else:
            frames_iter = self._frames_from_video(source)

        for frame_idx, frame in enumerate(frames_iter):
            if self.max_frames and frame_idx >= self.max_frames:
                break
            self._cache.process_frame(frame, frame_idx, store_frame=self.store_frames)
            if self.progress_callback:
                # We don't always know total upfront; pass 0 for total
                self.progress_callback(frame_idx + 1, 0)

        elapsed = time.time() - t0
        stats = self._cache.stats

        # Serialise outputs
        output_files += self._write_npz()
        output_files += self._write_json(source, elapsed)
        output_files += self._write_csv()

        return ProcessingResult(
            source=source,
            total_frames=stats.total_frames,
            anchor_frames=stats.anchor_frames,
            cached_frames=stats.cached_frames,
            cache_hit_rate=stats.cache_hit_rate,
            ram_reduction_pct=stats.ram_reduction_pct,
            bytes_saved=stats.bytes_saved,
            elapsed_seconds=elapsed,
            output_dir=str(self.output_dir),
            output_files=output_files,
            stats=stats,
        )

    def process_frames(
        self, frames: List[np.ndarray], source_label: str = "array"
    ) -> ProcessingResult:
        """Process an in-memory list of frames directly."""
        self._cache.reset()
        t0 = time.time()
        output_files: List[str] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx, frame in enumerate(frames):
            if self.max_frames and frame_idx >= self.max_frames:
                break
            self._cache.process_frame(frame, frame_idx, store_frame=self.store_frames)

        elapsed = time.time() - t0
        stats = self._cache.stats

        output_files += self._write_npz()
        output_files += self._write_json(source_label, elapsed)
        output_files += self._write_csv()

        return ProcessingResult(
            source=source_label,
            total_frames=stats.total_frames,
            anchor_frames=stats.anchor_frames,
            cached_frames=stats.cached_frames,
            cache_hit_rate=stats.cache_hit_rate,
            ram_reduction_pct=stats.ram_reduction_pct,
            bytes_saved=stats.bytes_saved,
            elapsed_seconds=elapsed,
            output_dir=str(self.output_dir),
            output_files=output_files,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Frame iterators
    # ------------------------------------------------------------------

    def _frames_from_video(self, path: str) -> Iterator[np.ndarray]:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required to read video files. "
                "Install it with: pip install opencv-python"
            ) from exc

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # cv2 uses BGR; convert to RGB
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    def _frames_from_dir(self, directory: str) -> Iterator[np.ndarray]:
        from PIL import Image as PILImage

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        paths = sorted(
            p for p in Path(directory).iterdir() if p.suffix.lower() in exts
        )
        for p in paths:
            img = PILImage.open(p).convert("RGB")
            yield np.array(img, dtype=np.uint8)

    def _synthetic_frames(
        self,
        n_frames: int = 200,
        height: int = 64,
        width: int = 64,
        scene_change_every: int = 40,
    ) -> Iterator[np.ndarray]:
        """
        Generate synthetic video frames for testing/demo.

        Creates a sequence with slow motion (high similarity) punctuated by
        occasional scene changes (low similarity) to showcase caching.
        """
        rng = np.random.default_rng(42)

        # Base scene
        base = rng.integers(30, 200, (height, width, 3), dtype=np.uint8)

        for i in range(n_frames):
            if self.max_frames and i >= self.max_frames:
                return

            if i % scene_change_every == 0 and i > 0:
                # Hard scene cut — new random base
                base = rng.integers(30, 200, (height, width, 3), dtype=np.uint8)
                frame = base.copy()
            else:
                # Small temporal perturbation (motion / noise)
                noise = rng.integers(-8, 9, (height, width, 3), dtype=np.int16)
                frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            yield frame

    # ------------------------------------------------------------------
    # Output writers
    # ------------------------------------------------------------------

    def _write_npz(self) -> List[str]:
        """Write compressed NPZ archive with cache map and anchor frames."""
        path = self.output_dir / "worldcache.npz"

        cache_map = self._cache.cache_map()
        frame_indices = np.array(sorted(cache_map.keys()), dtype=np.int32)
        anchor_refs = np.array([cache_map[i] for i in frame_indices], dtype=np.int32)

        data: dict = {
            "frame_indices": frame_indices,
            "anchor_refs": anchor_refs,
        }

        # Pack anchor frames into a single array (if stored)
        anchor_idxs = self._cache.anchor_indices()
        if anchor_idxs and self.store_frames:
            frames_list = [
                self._cache.get_anchor_frame(i)
                for i in anchor_idxs
                if self._cache.get_anchor_frame(i) is not None
            ]
            if frames_list:
                data["anchor_frame_indices"] = np.array(anchor_idxs, dtype=np.int32)
                data["anchor_frames"] = np.stack(frames_list, axis=0)

        np.savez_compressed(str(path), **data)
        return [str(path)]

    def _write_json(self, source: str, elapsed: float) -> List[str]:
        """Write human-readable JSON summary."""
        path = self.output_dir / "worldcache_stats.json"
        stats = self._cache.stats

        doc = {
            "source": source,
            "algorithm": self.algorithm.value,
            "similarity_threshold": self.similarity_threshold,
            "window_size": self.window_size,
            "hash_size": self.hash_size,
            "elapsed_seconds": round(elapsed, 3),
            "stats": {
                "total_frames": stats.total_frames,
                "anchor_frames": stats.anchor_frames,
                "cached_frames": stats.cached_frames,
                "cache_hit_rate": round(stats.cache_hit_rate, 4),
                "ram_reduction_pct": round(stats.ram_reduction_pct, 2),
                "bytes_saved": stats.bytes_saved,
                "bytes_used": stats.bytes_used,
            },
            "cache_map": {
                str(k): v for k, v in self._cache.cache_map().items()
            },
        }

        path.write_text(json.dumps(doc, indent=2))
        return [str(path)]

    def _write_csv(self) -> List[str]:
        """Write per-frame CSV for detailed analysis."""
        path = self.output_dir / "worldcache_frames.csv"

        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "frame_idx",
                    "is_anchor",
                    "anchor_idx",
                    "similarity",
                    "frame_hash",
                ],
            )
            writer.writeheader()
            for idx in sorted(self._cache.entries.keys()):
                e = self._cache.entries[idx]
                writer.writerow(
                    {
                        "frame_idx": e.frame_idx,
                        "is_anchor": int(e.is_anchor),
                        "anchor_idx": e.anchor_idx,
                        "similarity": round(e.similarity, 4),
                        "frame_hash": e.frame_hash,
                    }
                )

        return [str(path)]
