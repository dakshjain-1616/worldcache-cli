"""Tests for worldcache.cache — FrameCache and CacheStats."""

import numpy as np
import pytest

from worldcache.cache import CacheEntry, CacheStats, FrameCache
from worldcache.hasher import HashAlgorithm


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_frame(val: int, h: int = 32, w: int = 32) -> np.ndarray:
    """Solid-colour frame — easy to control similarity."""
    return np.full((h, w, 3), val, dtype=np.uint8)


def make_noisy_frame(base: np.ndarray, noise: int, rng) -> np.ndarray:
    n = rng.integers(-noise, noise + 1, base.shape, dtype=np.int16)
    return np.clip(base.astype(np.int16) + n, 0, 255).astype(np.uint8)


# ── CacheEntry ────────────────────────────────────────────────────────

def test_cache_entry_anchor():
    e = CacheEntry(frame_idx=0, is_anchor=True, anchor_idx=0, similarity=1.0, frame_hash=42)
    assert e.is_anchor
    assert not e.is_cached
    assert e.anchor_idx == 0


def test_cache_entry_cached():
    e = CacheEntry(frame_idx=5, is_anchor=False, anchor_idx=0, similarity=0.95, frame_hash=100)
    assert e.is_cached
    assert not e.is_anchor
    assert e.anchor_idx == 0


# ── CacheStats ────────────────────────────────────────────────────────

def test_cache_stats_hit_rate():
    s = CacheStats(total_frames=10, anchor_frames=2, cached_frames=8)
    assert s.cache_hit_rate == pytest.approx(0.8)
    assert s.ram_reduction_pct == pytest.approx(80.0)


def test_cache_stats_empty():
    s = CacheStats()
    assert s.cache_hit_rate == 0.0
    assert s.ram_reduction_pct == 0.0


def test_cache_stats_summary_contains_fields():
    s = CacheStats(total_frames=100, anchor_frames=10, cached_frames=90)
    summary = s.summary()
    assert "Total frames" in summary
    assert "Anchor frames" in summary
    assert "Cache hit rate" in summary
    assert "RAM reduction" in summary


# ── FrameCache — basic construction ───────────────────────────────────

def test_frame_cache_default_init():
    cache = FrameCache()
    assert cache.similarity_threshold > 0
    assert cache.window_size > 0
    assert len(cache.entries) == 0


# ── First frame is always an anchor ──────────────────────────────────

def test_first_frame_is_anchor():
    cache = FrameCache(algorithm=HashAlgorithm.DHASH, hash_size=8)
    frame = make_frame(100)
    entry = cache.process_frame(frame, frame_idx=0)
    assert entry.is_anchor
    assert entry.anchor_idx == 0
    assert cache.stats.anchor_frames == 1
    assert cache.stats.total_frames == 1


# ── Identical consecutive frames → cache hit ─────────────────────────

def test_identical_frames_cache_hit():
    cache = FrameCache(
        algorithm=HashAlgorithm.DHASH,
        similarity_threshold=0.85,
        hash_size=8,
    )
    frame = make_frame(150)
    cache.process_frame(frame, 0)
    entry = cache.process_frame(frame, 1)  # identical copy
    assert entry.is_cached
    assert entry.anchor_idx == 0
    assert cache.stats.cache_hit_rate > 0


# ── Very different frames → new anchor ───────────────────────────────

def test_different_frames_new_anchor():
    cache = FrameCache(
        algorithm=HashAlgorithm.DHASH,
        similarity_threshold=0.85,
        hash_size=8,
    )
    # Use textured (non-uniform) frames so dhash can detect the difference.
    # Solid-colour frames always produce hash=0 (no gradients), making them
    # indistinguishable by dhash regardless of brightness.
    rng = np.random.default_rng(999)
    f1 = rng.integers(0, 50, (32, 32, 3), dtype=np.uint8)    # dark textured
    f2 = rng.integers(200, 256, (32, 32, 3), dtype=np.uint8)  # bright textured

    cache.process_frame(f1, 0)
    entry = cache.process_frame(f2, 1)
    assert entry.is_anchor
    assert cache.stats.anchor_frames == 2


# ── Cache map integrity ───────────────────────────────────────────────

def test_cache_map_all_frames_present():
    cache = FrameCache(hash_size=8)
    for i in range(10):
        frame = make_frame(i * 25)
        cache.process_frame(frame, i)
    cmap = cache.cache_map()
    for i in range(10):
        assert i in cmap


# ── anchor_indices returns only anchors ───────────────────────────────

def test_anchor_indices_are_anchors():
    cache = FrameCache(hash_size=8)
    frames = [make_frame(10), make_frame(10), make_frame(200), make_frame(200)]
    for i, f in enumerate(frames):
        cache.process_frame(f, i)
    anchors = cache.anchor_indices()
    for idx in anchors:
        assert cache.entries[idx].is_anchor


# ── Stored anchor frames ──────────────────────────────────────────────

def test_anchor_frame_stored(rng):
    cache = FrameCache(hash_size=8, similarity_threshold=0.95)
    base = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    cache.process_frame(base, 0, store_frame=True)
    stored = cache.get_anchor_frame(0)
    assert stored is not None
    assert stored.shape == base.shape
    assert np.array_equal(stored, base)


# ── resolve_frame follows cache pointer ──────────────────────────────

def test_resolve_frame_returns_anchor_data():
    cache = FrameCache(hash_size=8, similarity_threshold=0.85)
    anchor = make_frame(100)
    identical = make_frame(100)
    cache.process_frame(anchor, 0)
    cache.process_frame(identical, 1)
    resolved = cache.resolve_frame(1)
    assert resolved is not None
    assert np.array_equal(resolved, anchor)


# ── Reset clears all state ────────────────────────────────────────────

def test_reset_clears_state():
    cache = FrameCache(hash_size=8)
    frame = make_frame(100)
    cache.process_frame(frame, 0)
    cache.reset()
    assert len(cache.entries) == 0
    assert cache.stats.total_frames == 0
    assert len(cache.anchor_indices()) == 0


# ── Window size limits comparison window ─────────────────────────────

def test_window_size_respected():
    cache = FrameCache(window_size=2, similarity_threshold=0.85, hash_size=8)
    # fill window with frames
    for i in range(5):
        cache.process_frame(make_frame(i * 20 + 10), i)
    # The internal window should never exceed window_size
    assert len(cache._window) <= 2


# ── get_entry returns correct entry ──────────────────────────────────

def test_get_entry_after_processing():
    cache = FrameCache(hash_size=8)
    frame = make_frame(77)
    cache.process_frame(frame, 42)
    entry = cache.get_entry(42)
    assert entry is not None
    assert entry.frame_idx == 42


def test_get_entry_missing_returns_none():
    cache = FrameCache(hash_size=8)
    assert cache.get_entry(999) is None


# ── Stats pixel tracking ──────────────────────────────────────────────

def test_stats_tracks_pixels_saved():
    cache = FrameCache(
        algorithm=HashAlgorithm.DHASH,
        similarity_threshold=0.85,
        hash_size=8,
    )
    f = make_frame(100)
    cache.process_frame(f, 0)
    cache.process_frame(f, 1)  # identical → cached
    assert cache.stats.total_pixels_saved > 0
