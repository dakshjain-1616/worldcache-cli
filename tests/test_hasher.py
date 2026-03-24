"""Tests for worldcache.hasher — temporal similarity hashing."""

import numpy as np
import pytest

from worldcache.hasher import HashAlgorithm, TemporalHasher


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def solid_frame():
    """A flat-colour frame (all pixels equal)."""
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def random_frame(rng):
    """A random RGB frame."""
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def similar_frame(random_frame, rng):
    """Frame very similar to random_frame (small noise)."""
    noise = rng.integers(-5, 6, random_frame.shape, dtype=np.int16)
    return np.clip(random_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def different_frame(rng):
    """A completely different random frame."""
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


# ── HashAlgorithm enum ────────────────────────────────────────────────

def test_algorithm_values():
    assert HashAlgorithm.DHASH.value == "dhash"
    assert HashAlgorithm.PHASH.value == "phash"
    assert HashAlgorithm.AHASH.value == "ahash"


def test_algorithm_from_string():
    assert HashAlgorithm("dhash") == HashAlgorithm.DHASH
    assert HashAlgorithm("phash") == HashAlgorithm.PHASH
    assert HashAlgorithm("ahash") == HashAlgorithm.AHASH


# ── TemporalHasher construction ───────────────────────────────────────

def test_hasher_default_algorithm():
    h = TemporalHasher()
    assert h.algorithm == HashAlgorithm.DHASH


def test_hasher_custom_algorithm():
    h = TemporalHasher(algorithm="phash", hash_size=8)
    assert h.algorithm == HashAlgorithm.PHASH
    assert h.hash_size == 8


# ── Hash return type ──────────────────────────────────────────────────

@pytest.mark.parametrize("algo", ["dhash", "phash", "ahash"])
def test_hash_returns_int(algo, random_frame):
    h = TemporalHasher(algorithm=algo, hash_size=8)
    result = h.hash(random_frame)
    assert isinstance(result, int)
    assert result >= 0


# ── Determinism ───────────────────────────────────────────────────────

@pytest.mark.parametrize("algo", ["dhash", "phash", "ahash"])
def test_hash_is_deterministic(algo, random_frame):
    h = TemporalHasher(algorithm=algo, hash_size=8)
    h1 = h.hash(random_frame)
    h2 = h.hash(random_frame)
    assert h1 == h2


# ── Identical frames → distance 0 ─────────────────────────────────────

@pytest.mark.parametrize("algo", ["dhash", "phash", "ahash"])
def test_identical_frames_zero_distance(algo, random_frame):
    h = TemporalHasher(algorithm=algo, hash_size=8)
    hash_val = h.hash(random_frame)
    assert TemporalHasher.hamming_distance(hash_val, hash_val) == 0


# ── Identical frames → similarity 1.0 ────────────────────────────────

@pytest.mark.parametrize("algo", ["dhash", "ahash"])
def test_identical_frames_max_similarity(algo, random_frame):
    h = TemporalHasher(algorithm=algo, hash_size=8)
    hv = h.hash(random_frame)
    assert h.similarity(hv, hv) == pytest.approx(1.0)


# ── Similar frames → high similarity ─────────────────────────────────

@pytest.mark.parametrize("algo", ["dhash", "ahash"])
def test_similar_frames_high_similarity(algo, random_frame, similar_frame):
    h = TemporalHasher(algorithm=algo, hash_size=8)
    h1 = h.hash(random_frame)
    h2 = h.hash(similar_frame)
    sim = h.similarity(h1, h2)
    assert sim > 0.70, f"Expected similar frames to have sim > 0.70, got {sim:.3f}"


# ── Completely different frames → lower similarity ─────────────────────

def test_different_frames_lower_similarity(random_frame, different_frame):
    h = TemporalHasher(algorithm="dhash", hash_size=8)
    h1 = h.hash(random_frame)
    h2 = h.hash(different_frame)
    sim_diff = h.similarity(h1, h2)
    sim_same = h.similarity(h1, h1)
    assert sim_same > sim_diff


# ── Grayscale input ───────────────────────────────────────────────────

def test_grayscale_input(rng):
    gray = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    h = TemporalHasher(algorithm="dhash", hash_size=8)
    result = h.hash(gray)
    assert isinstance(result, int)


# ── RGBA input ────────────────────────────────────────────────────────

def test_rgba_input(rng):
    rgba = rng.integers(0, 256, (64, 64, 4), dtype=np.uint8)
    h = TemporalHasher(algorithm="ahash", hash_size=8)
    result = h.hash(rgba)
    assert isinstance(result, int)


# ── Hamming distance ──────────────────────────────────────────────────

def test_hamming_distance_zero():
    assert TemporalHasher.hamming_distance(0b1010, 0b1010) == 0


def test_hamming_distance_all_bits():
    assert TemporalHasher.hamming_distance(0b0000, 0b1111) == 4


def test_hamming_distance_single_bit():
    assert TemporalHasher.hamming_distance(0b0001, 0b0000) == 1


# ── Similarity bounds ─────────────────────────────────────────────────

def test_similarity_bounded(rng):
    h = TemporalHasher(hash_size=8)
    for _ in range(20):
        f1 = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        f2 = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        sim = h.similarity(h.hash(f1), h.hash(f2))
        assert 0.0 <= sim <= 1.0
