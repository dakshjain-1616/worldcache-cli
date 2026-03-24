"""
WorldCache CLI - Content-aware frame caching for world model training.

Implements temporal similarity hashing to achieve 90% RAM reduction
when preprocessing video for world model training.
"""

__version__ = "1.0.0"
__author__ = "NEO"

from .cache import FrameCache, CacheEntry
from .hasher import TemporalHasher, HashAlgorithm
from .processor import VideoProcessor, ProcessingResult

__all__ = [
    "FrameCache",
    "CacheEntry",
    "TemporalHasher",
    "HashAlgorithm",
    "VideoProcessor",
    "ProcessingResult",
]
