"""Tests for worldcache.utils — helper functions."""

import numpy as np
import pytest

from worldcache.utils import (
    build_kv_reuse_schedule,
    estimate_ram_savings,
    format_bytes,
    generate_synthetic_video,
)


def test_generate_synthetic_video_length():
    frames = generate_synthetic_video(n_frames=30)
    assert len(frames) == 30


def test_generate_synthetic_video_shape():
    frames = generate_synthetic_video(n_frames=5, height=48, width=64)
    for f in frames:
        assert f.shape == (48, 64, 3)
        assert f.dtype == np.uint8


def test_generate_synthetic_video_deterministic():
    f1 = generate_synthetic_video(n_frames=10, seed=7)
    f2 = generate_synthetic_video(n_frames=10, seed=7)
    for a, b in zip(f1, f2):
        assert np.array_equal(a, b)


def test_generate_synthetic_video_scene_changes():
    # Scene changes should create visible pixel value differences
    frames = generate_synthetic_video(n_frames=100, scene_change_every=20, seed=0)
    diffs = []
    for i in range(1, len(frames)):
        d = np.abs(frames[i].astype(int) - frames[i - 1].astype(int)).mean()
        diffs.append(d)
    # At least one large diff (scene change)
    assert max(diffs) > 20


def test_estimate_ram_savings_full_cache():
    result = estimate_ram_savings(
        total_frames=100, anchor_frames=10, frame_shape=(64, 64, 3)
    )
    assert result["pct_reduction"] == pytest.approx(90.0)
    assert result["bytes_saved"] > 0


def test_estimate_ram_savings_no_cache():
    result = estimate_ram_savings(
        total_frames=100, anchor_frames=100, frame_shape=(64, 64, 3)
    )
    assert result["pct_reduction"] == pytest.approx(0.0)
    assert result["bytes_saved"] == 0


def test_estimate_ram_savings_zero_frames():
    result = estimate_ram_savings(total_frames=0, anchor_frames=0)
    assert result["pct_reduction"] == 0.0


def test_build_kv_reuse_schedule_anchors():
    cache_map = {0: 0, 1: 0, 2: 2, 3: 2, 4: 4}
    schedule = build_kv_reuse_schedule(cache_map)
    assert len(schedule) == 5
    # Anchor frames have reuse_from = None
    anchor_entries = [(i, r) for i, r in schedule if r is None]
    assert len(anchor_entries) == 3  # frames 0, 2, 4


def test_build_kv_reuse_schedule_cached_points_to_anchor():
    cache_map = {0: 0, 1: 0, 2: 0}
    schedule = build_kv_reuse_schedule(cache_map)
    # frame 0 → None (anchor); frames 1, 2 → 0
    assert schedule[0] == (0, None)
    assert schedule[1] == (1, 0)
    assert schedule[2] == (2, 0)


def test_build_kv_reuse_schedule_sorted():
    cache_map = {5: 5, 2: 2, 0: 0, 3: 2}
    schedule = build_kv_reuse_schedule(cache_map)
    indices = [s[0] for s in schedule]
    assert indices == sorted(indices)


def test_format_bytes_bytes():
    assert format_bytes(512) == "512.0 B"


def test_format_bytes_kb():
    assert "KB" in format_bytes(2048)


def test_format_bytes_mb():
    assert "MB" in format_bytes(1024 * 1024 * 5)
