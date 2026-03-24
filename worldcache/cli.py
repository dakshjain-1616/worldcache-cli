"""
WorldCache CLI entry points.

Commands
--------
  worldcache process   Process a video file or frame directory
  worldcache inspect   Print cache statistics from a previous run
  worldcache demo      Run a quick synthetic demo
  worldcache benchmark Compare all hash algorithms on a source
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import click

from .hasher import HashAlgorithm
from .processor import VideoProcessor

# ── Colour helpers (graceful fallback if rich not installed) ──────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

    _RICH = True
except ImportError:  # pragma: no cover
    _RICH = False


def _get_console():
    """Create a Console bound to the *current* sys.stdout.

    Creating Console() here (rather than at module import time) ensures that
    Click's test runner (CliRunner) can capture rich output, because CliRunner
    patches sys.stdout before invoking commands.
    """
    import sys
    return Console(file=sys.stdout, highlight=False)  # type: ignore


def _print(msg: str, style: str = "") -> None:
    if _RICH:
        _get_console().print(msg, style=style)  # type: ignore
    else:
        click.echo(msg)


def _print_result_table(result) -> None:
    if _RICH:
        console = _get_console()
        table = Table(title="WorldCache Results", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim", width=28)
        table.add_column("Value", justify="right")
        table.add_row("Source", result.source)
        table.add_row("Total frames", f"{result.total_frames:,}")
        table.add_row("Anchor frames", f"{result.anchor_frames:,}")
        table.add_row("Cached frames", f"{result.cached_frames:,}")
        table.add_row("Cache hit rate", f"{result.cache_hit_rate:.1%}")
        table.add_row(
            "RAM reduction",
            f"[green bold]{result.ram_reduction_pct:.1f}%[/green bold]",
        )
        mb_saved = result.bytes_saved / (1024 ** 2) if result.bytes_saved else 0
        table.add_row("Bytes saved", f"{mb_saved:.2f} MB")
        table.add_row("Elapsed", f"{result.elapsed_seconds:.2f}s")
        table.add_row("Output dir", result.output_dir)
        console.print(table)  # type: ignore
    else:
        click.echo(result.stats.summary())
        click.echo(f"Elapsed: {result.elapsed_seconds:.2f}s")
        click.echo(f"Output: {result.output_dir}")


# ── CLI group ─────────────────────────────────────────────────────────
@click.group()
@click.version_option(version="1.0.0", prog_name="worldcache")
def cli() -> None:
    """WorldCache CLI - Content-aware frame caching for world model training.

    Implements temporal similarity hashing to achieve up to 90% RAM reduction
    when pre-processing video datasets for world model training.
    """


# ── process ───────────────────────────────────────────────────────────
@cli.command()
@click.argument("source", default="synthetic")
@click.option(
    "--output-dir", "-o",
    default=os.getenv("WORLDCACHE_OUTPUT_DIR", "worldcache_output"),
    show_default=True,
    help="Directory to write output files.",
)
@click.option(
    "--algorithm", "-a",
    type=click.Choice(["dhash", "phash", "ahash"], case_sensitive=False),
    default=os.getenv("WORLDCACHE_ALGORITHM", "dhash"),
    show_default=True,
    help="Hash algorithm for temporal similarity.",
)
@click.option(
    "--threshold", "-t",
    type=float,
    default=float(os.getenv("WORLDCACHE_SIMILARITY_THRESHOLD", "0.92")),
    show_default=True,
    help="Similarity threshold [0, 1]. Higher = fewer anchors.",
)
@click.option(
    "--window", "-w",
    type=int,
    default=int(os.getenv("WORLDCACHE_WINDOW_SIZE", "32")),
    show_default=True,
    help="Sliding window size for anchor comparison.",
)
@click.option(
    "--hash-size",
    type=int,
    default=int(os.getenv("WORLDCACHE_HASH_SIZE", "16")),
    show_default=True,
    help="Hash grid size. Hash length = hash_size².",
)
@click.option(
    "--max-frames", "-n",
    type=int,
    default=int(os.getenv("WORLDCACHE_MAX_FRAMES", "0")),
    show_default=True,
    help="Limit number of frames (0 = unlimited).",
)
@click.option(
    "--no-store-frames",
    is_flag=True,
    default=False,
    help="Skip storing anchor frames in memory (hash-only mode).",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    default=False,
    help="Suppress progress output.",
)
def process(
    source: str,
    output_dir: str,
    algorithm: str,
    threshold: float,
    window: int,
    hash_size: int,
    max_frames: int,
    no_store_frames: bool,
    quiet: bool,
) -> None:
    """Process a video SOURCE and write cache outputs.

    SOURCE can be:
    \b
      - Path to a video file  (e.g. video.mp4)
      - Path to a directory of image frames
      - "synthetic"           (built-in test generator)
    """
    if not quiet:
        _print(f"[bold cyan]WorldCache[/bold cyan] processing: [yellow]{source}[/yellow]")

    callback = None
    if not quiet and not _RICH:
        def callback(done: int, total: int) -> None:
            print(f"\r  Frames processed: {done}", end="", flush=True)

    proc = VideoProcessor(
        output_dir=output_dir,
        algorithm=algorithm,
        similarity_threshold=threshold,
        window_size=window,
        hash_size=hash_size,
        max_frames=max_frames,
        store_frames=not no_store_frames,
        progress_callback=callback,
    )

    if not quiet and _RICH:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}[/cyan] frames"),
            TimeElapsedColumn(),
            console=_get_console(),
            transient=True,
        ) as progress:
            task = progress.add_task("Processing frames…", total=None)

            def rich_callback(done: int, _total: int) -> None:
                progress.update(task, completed=done)

            proc.progress_callback = rich_callback
            result = proc.process(source)
    else:
        result = proc.process(source)
        if not quiet:
            print()  # newline after inline counter

    if not quiet:
        _print_result_table(result)
        _print(
            f"\n[bold green]Done![/bold green] Output written to [yellow]{output_dir}[/yellow]"
        )


# ── inspect ───────────────────────────────────────────────────────────
@cli.command()
@click.argument(
    "stats_file",
    default=os.path.join(
        os.getenv("WORLDCACHE_OUTPUT_DIR", "worldcache_output"),
        "worldcache_stats.json",
    ),
)
def inspect(stats_file: str) -> None:
    """Inspect statistics from a previous WorldCache run.

    STATS_FILE defaults to worldcache_output/worldcache_stats.json.
    """
    p = Path(stats_file)
    if not p.exists():
        click.echo(f"File not found: {stats_file}", err=True)
        sys.exit(1)

    doc = json.loads(p.read_text())
    stats = doc.get("stats", {})

    if _RICH:
        table = Table(title=f"Inspection: {stats_file}", header_style="bold magenta")
        table.add_column("Key", style="dim")
        table.add_column("Value", justify="right")
        for k, v in stats.items():
            table.add_row(str(k), str(v))
        _get_console().print(table)  # type: ignore
    else:
        for k, v in stats.items():
            click.echo(f"  {k:<28}: {v}")


# ── demo ──────────────────────────────────────────────────────────────
@cli.command()
@click.option(
    "--frames", "-n", type=int, default=200, help="Number of synthetic frames."
)
@click.option(
    "--output-dir", "-o",
    default=os.getenv("WORLDCACHE_OUTPUT_DIR", "worldcache_output"),
    help="Output directory.",
)
def demo(frames: int, output_dir: str) -> None:
    """Run a synthetic demo showcasing WorldCache caching efficiency."""
    _print("[bold cyan]WorldCache Demo[/bold cyan] — synthetic video with scene changes")

    proc = VideoProcessor(
        output_dir=output_dir,
        algorithm="dhash",
        similarity_threshold=0.92,
        window_size=32,
        max_frames=frames,
    )
    result = proc.process("synthetic")
    _print_result_table(result)
    _print(
        "\n[dim]Outputs written to[/dim] [yellow]{output}[/yellow]".format(
            output=output_dir
        )
    )


# ── benchmark ─────────────────────────────────────────────────────────
@cli.command()
@click.argument("source", default="synthetic")
@click.option(
    "--frames", "-n", type=int, default=100, help="Number of frames to benchmark."
)
def benchmark(source: str, frames: int) -> None:
    """Benchmark all hash algorithms on SOURCE and compare results."""
    _print("[bold cyan]WorldCache Benchmark[/bold cyan]")

    import tempfile

    results = []
    for algo in ["dhash", "phash", "ahash"]:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = VideoProcessor(
                output_dir=tmpdir,
                algorithm=algo,
                max_frames=frames,
            )
            t0 = time.time()
            result = proc.process(source)
            elapsed = time.time() - t0
            results.append(
                {
                    "algorithm": algo,
                    "hit_rate": result.cache_hit_rate,
                    "ram_reduction": result.ram_reduction_pct,
                    "fps": frames / elapsed if elapsed > 0 else 0,
                }
            )

    if _RICH:
        table = Table(title="Algorithm Benchmark", header_style="bold yellow")
        table.add_column("Algorithm", style="cyan")
        table.add_column("Hit Rate", justify="right")
        table.add_column("RAM Reduction", justify="right")
        table.add_column("Speed (fps)", justify="right")
        for r in results:
            table.add_row(
                r["algorithm"],
                f"{r['hit_rate']:.1%}",
                f"{r['ram_reduction']:.1f}%",
                f"{r['fps']:.0f}",
            )
        _get_console().print(table)  # type: ignore
    else:
        click.echo(f"{'Algorithm':<12} {'Hit Rate':>10} {'RAM Reduction':>14} {'fps':>8}")
        for r in results:
            click.echo(
                f"{r['algorithm']:<12} {r['hit_rate']:>9.1%} "
                f"{r['ram_reduction']:>13.1f}% {r['fps']:>7.0f}"
            )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
