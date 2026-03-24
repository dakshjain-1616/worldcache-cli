"""Tests for worldcache.cli — Click command interface."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from worldcache.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_out(tmp_path):
    return str(tmp_path / "out")


# ── worldcache --help ─────────────────────────────────────────────────

def test_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "WorldCache" in result.output


def test_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "1.0.0" in result.output


# ── worldcache process ────────────────────────────────────────────────

def test_process_synthetic(runner, tmp_out):
    result = runner.invoke(
        cli,
        ["process", "synthetic", "--output-dir", tmp_out, "--max-frames", "50"],
    )
    assert result.exit_code == 0, result.output


def test_process_creates_output_files(runner, tmp_out):
    runner.invoke(
        cli,
        ["process", "synthetic", "--output-dir", tmp_out, "--max-frames", "30"],
    )
    out = Path(tmp_out)
    assert (out / "worldcache.npz").exists()
    assert (out / "worldcache_stats.json").exists()
    assert (out / "worldcache_frames.csv").exists()


def test_process_quiet_flag(runner, tmp_out):
    result = runner.invoke(
        cli,
        [
            "process", "synthetic",
            "--output-dir", tmp_out,
            "--max-frames", "20",
            "--quiet",
        ],
    )
    assert result.exit_code == 0


def test_process_algorithm_option(runner, tmp_out):
    result = runner.invoke(
        cli,
        [
            "process", "synthetic",
            "--output-dir", tmp_out,
            "--algorithm", "ahash",
            "--max-frames", "20",
        ],
    )
    assert result.exit_code == 0


def test_process_threshold_option(runner, tmp_out):
    result = runner.invoke(
        cli,
        [
            "process", "synthetic",
            "--output-dir", tmp_out,
            "--threshold", "0.80",
            "--max-frames", "20",
        ],
    )
    assert result.exit_code == 0
    stats_doc = json.loads((Path(tmp_out) / "worldcache_stats.json").read_text())
    assert stats_doc["similarity_threshold"] == pytest.approx(0.80)


def test_process_no_store_frames(runner, tmp_out):
    result = runner.invoke(
        cli,
        [
            "process", "synthetic",
            "--output-dir", tmp_out,
            "--no-store-frames",
            "--max-frames", "20",
        ],
    )
    assert result.exit_code == 0


# ── worldcache demo ───────────────────────────────────────────────────

def test_demo_command(runner, tmp_out):
    result = runner.invoke(
        cli, ["demo", "--frames", "40", "--output-dir", tmp_out]
    )
    assert result.exit_code == 0


# ── worldcache inspect ────────────────────────────────────────────────

def test_inspect_command(runner, tmp_out):
    # First generate a stats file
    runner.invoke(
        cli,
        ["process", "synthetic", "--output-dir", tmp_out, "--max-frames", "30"],
    )
    stats_file = str(Path(tmp_out) / "worldcache_stats.json")
    result = runner.invoke(cli, ["inspect", stats_file])
    assert result.exit_code == 0


def test_inspect_missing_file(runner):
    result = runner.invoke(cli, ["inspect", "/nonexistent/path/stats.json"])
    assert result.exit_code != 0


# ── worldcache benchmark ──────────────────────────────────────────────

def test_benchmark_command(runner):
    result = runner.invoke(
        cli, ["benchmark", "synthetic", "--frames", "30"]
    )
    assert result.exit_code == 0
    # All three algorithms should appear
    for algo in ("dhash", "phash", "ahash"):
        assert algo in result.output

