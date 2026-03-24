"""Tests for worldcache.processor — VideoProcessor and ProcessingResult."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from worldcache.processor import VideoProcessor
from worldcache.utils import generate_synthetic_video, load_npz_cache, reconstruct_cache_map


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_output(tmp_path):
    return str(tmp_path / "output")


@pytest.fixture
def simple_frames():
    """50 frames with a scene change at frame 25."""
    return generate_synthetic_video(n_frames=50, height=32, width=32, scene_change_every=25)


# ── VideoProcessor construction ───────────────────────────────────────

def test_processor_default_construction():
    proc = VideoProcessor()
    assert proc.similarity_threshold > 0
    assert proc.window_size > 0


def test_processor_custom_params(tmp_output):
    proc = VideoProcessor(
        output_dir=tmp_output,
        algorithm="phash",
        similarity_threshold=0.88,
        window_size=16,
        hash_size=8,
        max_frames=20,
    )
    assert proc.similarity_threshold == pytest.approx(0.88)
    assert proc.window_size == 16
    assert proc.max_frames == 20


# ── process_frames returns valid ProcessingResult ─────────────────────

def test_process_frames_returns_result(tmp_output, simple_frames):
    proc = VideoProcessor(output_dir=tmp_output, hash_size=8)
    result = proc.process_frames(simple_frames, source_label="test")
    assert result.total_frames == len(simple_frames)
    assert result.anchor_frames >= 1
    assert result.cached_frames >= 0
    assert result.anchor_frames + result.cached_frames == result.total_frames


# ── Cache hit rate is sane ────────────────────────────────────────────

def test_cache_hit_rate_is_sane(tmp_output, simple_frames):
    proc = VideoProcessor(
        output_dir=tmp_output,
        algorithm="dhash",
        similarity_threshold=0.92,
        hash_size=8,
    )
    result = proc.process_frames(simple_frames)
    # With slow motion + small noise the hit rate should be significant
    assert 0.0 <= result.cache_hit_rate <= 1.0


# ── Highly similar frames give high hit rate ─────────────────────────

def test_nearly_identical_frames_high_hit_rate(tmp_output):
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    # 90 near-identical + 10 different
    frames = []
    for _ in range(90):
        n = rng.integers(-3, 4, base.shape, dtype=np.int16)
        frames.append(np.clip(base.astype(np.int16) + n, 0, 255).astype(np.uint8))
    for _ in range(10):
        frames.append(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))

    proc = VideoProcessor(output_dir=tmp_output, algorithm="dhash", similarity_threshold=0.90, hash_size=8)
    result = proc.process_frames(frames)
    assert result.cache_hit_rate > 0.5


# ── Synthetic source works ────────────────────────────────────────────

def test_process_synthetic_source(tmp_output):
    proc = VideoProcessor(output_dir=tmp_output, max_frames=50, hash_size=8)
    result = proc.process("synthetic")
    assert result.total_frames == 50
    assert result.source == "synthetic"


# ── max_frames limits processing ─────────────────────────────────────

def test_max_frames_limit(tmp_output, simple_frames):
    proc = VideoProcessor(output_dir=tmp_output, max_frames=10, hash_size=8)
    result = proc.process_frames(simple_frames)
    assert result.total_frames == 10


# ── Output files are created ─────────────────────────────────────────

def test_output_files_created(tmp_output, simple_frames):
    proc = VideoProcessor(output_dir=tmp_output, hash_size=8)
    result = proc.process_frames(simple_frames)
    out_dir = Path(tmp_output)
    assert (out_dir / "worldcache.npz").exists()
    assert (out_dir / "worldcache_stats.json").exists()
    assert (out_dir / "worldcache_frames.csv").exists()


# ── NPZ contents are valid ────────────────────────────────────────────

def test_npz_contents(tmp_output, simple_frames):
    proc = VideoProcessor(output_dir=tmp_output, hash_size=8)
    result = proc.process_frames(simple_frames)
    data = load_npz_cache(str(Path(tmp_output) / "worldcache.npz"))
    assert "frame_indices" in data
    assert "anchor_refs" in data
    assert len(data["frame_indices"]) == len(simple_frames)


# ── Cache map from NPZ is consistent ─────────────────────────────────

def test_npz_cache_map_consistent(tmp_output, simple_frames):
    proc = VideoProcessor(output_dir=tmp_output, hash_size=8)
    result = proc.process_frames(simple_frames)
    data = load_npz_cache(str(Path(tmp_output) / "worldcache.npz"))
    cmap = reconstruct_cache_map(data)
    assert len(cmap) == len(simple_frames)
    for idx, anchor in cmap.items():
        assert 0 <= anchor < len(simple_frames)


# ── JSON stats file is valid ──────────────────────────────────────────

def test_json_stats_valid(tmp_output, simple_frames):
    proc = VideoProcessor(output_dir=tmp_output, hash_size=8)
    result = proc.process_frames(simple_frames)
    json_path = Path(tmp_output) / "worldcache_stats.json"
    doc = json.loads(json_path.read_text())
    assert "stats" in doc
    stats = doc["stats"]
    assert "total_frames" in stats
    assert stats["total_frames"] == len(simple_frames)
    assert "cache_hit_rate" in stats
    assert "ram_reduction_pct" in stats


# ── CSV contains all frames ───────────────────────────────────────────

def test_csv_row_count(tmp_output, simple_frames):
    import csv
    proc = VideoProcessor(output_dir=tmp_output, hash_size=8)
    proc.process_frames(simple_frames)
    csv_path = Path(tmp_output) / "worldcache_frames.csv"
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(simple_frames)
    assert "frame_idx" in rows[0]
    assert "is_anchor" in rows[0]
    assert "similarity" in rows[0]


# ── elapsed_seconds is positive ───────────────────────────────────────

def test_elapsed_positive(tmp_output, simple_frames):
    proc = VideoProcessor(output_dir=tmp_output, hash_size=8)
    result = proc.process_frames(simple_frames)
    assert result.elapsed_seconds > 0


# ── to_dict round-trips cleanly ──────────────────────────────────────

def test_result_to_dict(tmp_output, simple_frames):
    proc = VideoProcessor(output_dir=tmp_output, hash_size=8)
    result = proc.process_frames(simple_frames)
    d = result.to_dict()
    assert isinstance(d, dict)
    assert d["total_frames"] == len(simple_frames)
    assert 0.0 <= d["cache_hit_rate"] <= 1.0


# ── scene_change_every affects synthetic cache hit rate ───────────────

def test_scene_change_every_affects_hit_rate(tmp_output):
    """Slow motion (rare scene changes) should give higher hit rate than rapid cuts."""
    import tempfile, os

    slow_dir = os.path.join(tmp_output, "slow")
    fast_dir = os.path.join(tmp_output, "fast")

    proc_slow = VideoProcessor(
        output_dir=slow_dir, hash_size=8, max_frames=60, scene_change_every=60
    )
    proc_fast = VideoProcessor(
        output_dir=fast_dir, hash_size=8, max_frames=60, scene_change_every=5
    )
    result_slow = proc_slow.process("synthetic")
    result_fast = proc_fast.process("synthetic")
    # Slow motion should cache more frames than rapid cuts
    assert result_slow.cache_hit_rate >= result_fast.cache_hit_rate
