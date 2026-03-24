"""
Temporal similarity hashing for WorldCache.

Implements three hashing algorithms:
  - dHash (difference hash): captures spatial gradients, great for motion
  - pHash (perceptual hash): DCT-based, robust to minor perturbations
  - aHash (average hash): fast baseline using mean thresholding

All hashes are returned as Python ints; distance is measured via Hamming distance.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Union

import numpy as np
from PIL import Image


class HashAlgorithm(str, Enum):
    DHASH = "dhash"
    PHASH = "phash"
    AHASH = "ahash"


# Default hash size (bits = hash_size²)
DEFAULT_HASH_SIZE = int(os.getenv("WORLDCACHE_HASH_SIZE", "16"))


class TemporalHasher:
    """Computes perceptual hashes for video frames and measures temporal similarity."""

    def __init__(
        self,
        algorithm: Union[HashAlgorithm, str] = HashAlgorithm.DHASH,
        hash_size: int = DEFAULT_HASH_SIZE,
    ) -> None:
        self.algorithm = HashAlgorithm(algorithm)
        self.hash_size = hash_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def hash(self, frame: np.ndarray) -> int:
        """Compute a perceptual hash for a single video frame (H×W×C or H×W)."""
        gray = self._to_gray(frame)
        if self.algorithm == HashAlgorithm.DHASH:
            return self._dhash(gray)
        elif self.algorithm == HashAlgorithm.PHASH:
            return self._phash(gray)
        else:
            return self._ahash(gray)

    def similarity(self, hash1: int, hash2: int) -> float:
        """Return similarity in [0, 1] where 1.0 means identical."""
        max_bits = self.hash_size * self.hash_size
        dist = self.hamming_distance(hash1, hash2)
        return 1.0 - dist / max_bits

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Bit-level Hamming distance between two hashes."""
        return bin(hash1 ^ hash2).count("1")

    # ------------------------------------------------------------------
    # Hash algorithms
    # ------------------------------------------------------------------

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame array to grayscale uint8."""
        if frame.ndim == 2:
            return frame.astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 4:
            # RGBA → RGB first
            frame = frame[:, :, :3]
        if frame.ndim == 3:
            # Weighted luminance conversion (BT.601)
            gray = (
                0.299 * frame[:, :, 0].astype(np.float32)
                + 0.587 * frame[:, :, 1].astype(np.float32)
                + 0.114 * frame[:, :, 2].astype(np.float32)
            )
            return np.clip(gray, 0, 255).astype(np.uint8)
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    def _dhash(self, gray: np.ndarray) -> int:
        """
        Difference hash — compares each pixel to its right neighbor.
        Excellent for capturing motion between consecutive frames.
        Hash length = hash_size * hash_size bits.
        """
        img = Image.fromarray(gray)
        img = img.resize((self.hash_size + 1, self.hash_size), Image.LANCZOS)
        pixels = np.array(img, dtype=np.int32)
        # Horizontal differences
        diff = pixels[:, 1:] > pixels[:, :-1]  # (H, W)
        return self._bits_to_int(diff.flatten())

    def _phash(self, gray: np.ndarray) -> int:
        """
        Perceptual hash using 2-D FFT magnitude (approximates DCT low-frequency block).
        Robust against minor brightness/contrast changes.
        The FFT magnitude spectrum captures the same low-frequency dominance as DCT-based pHash.
        """
        dct_size = self.hash_size * 4  # oversample for better frequency resolution
        img = Image.fromarray(gray)
        img = img.resize((dct_size, dct_size), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float64)

        # 2-D FFT magnitude — low-frequency block occupies the top-left corner
        fft_mag = np.abs(np.fft.fft2(pixels))
        # Take top-left hash_size x hash_size low-frequency region
        low = fft_mag[: self.hash_size, : self.hash_size]
        # Exclude DC component (index 0,0) for mean threshold
        flat = low.flatten()
        mean_val = (flat.sum() - flat[0]) / (len(flat) - 1)
        bits = flat > mean_val
        return self._bits_to_int(bits)

    def _ahash(self, gray: np.ndarray) -> int:
        """
        Average hash — fastest algorithm, uses mean pixel threshold.
        """
        img = Image.fromarray(gray)
        img = img.resize((self.hash_size, self.hash_size), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float32)
        mean_val = pixels.mean()
        bits = pixels > mean_val
        return self._bits_to_int(bits.flatten())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bits_to_int(bits: np.ndarray) -> int:
        """Pack boolean array into a Python int (MSB first)."""
        val = 0
        for b in bits:
            val = (val << 1) | int(bool(b))
        return val
