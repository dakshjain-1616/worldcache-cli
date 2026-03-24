"""
WorldCache CLI — Interactive demo script.

Demonstrates content-aware frame caching with temporal similarity hashing,
achieving up to 90% RAM reduction for world model training.

Run:
    python demo.py

No API keys or real video files required. All frames are generated synthetically.
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# ── Banner ─────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════╗
║         WorldCache CLI — Frame Caching Demo              ║
║   Pre-process video for world models at 90% less RAM     ║
╚══════════════════════════════════════════════════════════╝
"""

# ── Colour helpers ─────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich import print as rprint

    console = Console()
    RICH = True
except ImportError:
    RICH = False
    console = None


def cprint(msg: str, style: str = "") -> None:
    if RICH:
        console.print(msg, style=style)
    else:
        # Strip rich markup
        import re
        clean = re.sub(r"\[/?[^\[\]]*\]", "", msg)
        print(clean)


# ── Demo scenarios ─────────────────────────────────────────────────────

def run_demo_scenario(
    name: str,
    description: str,
    n_frames: int,
    scene_change_every: int,
    similarity_threshold: float,
    algorithm: str,
    output_dir: str,
) -> dict:
    """Run a single demo scenario and return the results dict."""
    from worldcache.processor import VideoProcessor

    proc = VideoProcessor(
        output_dir=output_dir,
        algorithm=algorithm,
        similarity_threshold=similarity_threshold,
        window_size=int(os.getenv("WORLDCACHE_WINDOW_SIZE", "32")),
        hash_size=int(os.getenv("WORLDCACHE_HASH_SIZE", "16")),
        max_frames=n_frames,
        scene_change_every=scene_change_every,
    )

    cprint(f"\n[bold cyan]Scenario:[/bold cyan] {name}")
    cprint(f"[dim]{description}[/dim]")

    t0 = time.time()
    result = proc.process("synthetic")
    elapsed = time.time() - t0

    return {
        "name": name,
        "total_frames": result.total_frames,
        "anchor_frames": result.anchor_frames,
        "cached_frames": result.cached_frames,
        "hit_rate": result.cache_hit_rate,
        "ram_reduction_pct": result.ram_reduction_pct,
        "bytes_saved": result.bytes_saved,
        "elapsed": elapsed,
        "output_dir": output_dir,
    }


def print_scenario_result(r: dict) -> None:
    mb_saved = r["bytes_saved"] / (1024 ** 2) if r["bytes_saved"] else 0

    if RICH:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="dim", width=22)
        table.add_column("Value", justify="right")
        table.add_row("Total frames", str(r["total_frames"]))
        table.add_row("Anchor frames", str(r["anchor_frames"]))
        table.add_row("Cached frames", str(r["cached_frames"]))
        table.add_row("Cache hit rate", f"{r['hit_rate']:.1%}")
        table.add_row(
            "RAM reduction",
            f"[bold green]{r['ram_reduction_pct']:.1f}%[/bold green]",
        )
        if mb_saved > 0:
            table.add_row("Bytes saved", f"{mb_saved:.2f} MB")
        table.add_row("Speed", f"{r['total_frames']/r['elapsed']:.0f} fps")
        console.print(table)
    else:
        print(f"  Total frames    : {r['total_frames']}")
        print(f"  Anchor frames   : {r['anchor_frames']}")
        print(f"  Cached frames   : {r['cached_frames']}")
        print(f"  Cache hit rate  : {r['hit_rate']:.1%}")
        print(f"  RAM reduction   : {r['ram_reduction_pct']:.1f}%")
        if mb_saved > 0:
            print(f"  Bytes saved     : {mb_saved:.2f} MB")
        print(f"  Speed           : {r['total_frames']/r['elapsed']:.0f} fps")


def print_comparison_table(results: list) -> None:
    cprint("\n[bold]Algorithm Comparison[/bold]")

    if RICH:
        table = Table(header_style="bold yellow", show_lines=False)
        table.add_column("Scenario", style="cyan", min_width=26)
        table.add_column("Frames", justify="right")
        table.add_column("Hit Rate", justify="right")
        table.add_column("RAM Reduction", justify="right")
        table.add_column("fps", justify="right")
        for r in results:
            fps = r["total_frames"] / r["elapsed"] if r["elapsed"] > 0 else 0
            table.add_row(
                r["name"],
                str(r["total_frames"]),
                f"{r['hit_rate']:.1%}",
                f"[green]{r['ram_reduction_pct']:.1f}%[/green]",
                f"{fps:.0f}",
            )
        console.print(table)
    else:
        header = f"{'Scenario':<26} {'Frames':>7} {'Hit Rate':>10} {'RAM Saved':>12} {'fps':>6}"
        print(header)
        print("-" * len(header))
        for r in results:
            fps = r["total_frames"] / r["elapsed"] if r["elapsed"] > 0 else 0
            print(
                f"{r['name']:<26} {r['total_frames']:>7} "
                f"{r['hit_rate']:>9.1%} {r['ram_reduction_pct']:>11.1f}% {fps:>5.0f}"
            )


def demo_kv_reuse_schedule(output_dir: str) -> None:
    """Show the KV-cache reuse schedule generated from the cache map."""
    from worldcache.utils import (
        build_kv_reuse_schedule,
        load_npz_cache,
        reconstruct_cache_map,
    )

    npz_path = Path(output_dir) / "worldcache.npz"
    if not npz_path.exists():
        return

    data = load_npz_cache(str(npz_path))
    cmap = reconstruct_cache_map(data)
    schedule = build_kv_reuse_schedule(cmap)

    cprint("\n[bold cyan]KV-Cache Reuse Schedule (first 20 frames)[/bold cyan]")
    cprint("[dim]Shows which frames can reuse KV-cache from an anchor frame.[/dim]\n")

    if RICH:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Frame", justify="right", width=7)
        table.add_column("Type", width=10)
        table.add_column("Reuse from", justify="right", width=12)
        for frame_idx, reuse_from in schedule[:20]:
            if reuse_from is None:
                table.add_row(str(frame_idx), "[green]ANCHOR[/green]", "—")
            else:
                table.add_row(
                    str(frame_idx),
                    "[yellow]CACHED[/yellow]",
                    f"frame {reuse_from}",
                )
        console.print(table)
    else:
        print(f"  {'Frame':>6}  {'Type':<8}  {'Reuse from':>12}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*12}")
        for frame_idx, reuse_from in schedule[:20]:
            rtype = "ANCHOR" if reuse_from is None else "CACHED"
            rfrom = "—" if reuse_from is None else f"frame {reuse_from}"
            print(f"  {frame_idx:>6}  {rtype:<8}  {rfrom:>12}")


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    print(BANNER)
    cprint(
        "[bold green]WorldCache[/bold green] pre-processes video frames using "
        "[bold]temporal similarity hashing[/bold]."
    )
    cprint("Similar consecutive frames share the same KV-cache slot → massive RAM savings.\n")

    base_output = os.getenv("WORLDCACHE_OUTPUT_DIR", "worldcache_output")
    Path(base_output).mkdir(parents=True, exist_ok=True)

    results = []

    # ── Scenario 1: Slow-motion video (many similar frames) ──────────
    out1 = str(Path(base_output) / "scenario_slow_motion")
    r1 = run_demo_scenario(
        name="Slow motion (high similarity)",
        description="Frames change very little between steps — like a slow pan.",
        n_frames=200,
        scene_change_every=100,
        similarity_threshold=float(os.getenv("WORLDCACHE_SIMILARITY_THRESHOLD", "0.92")),
        algorithm="dhash",
        output_dir=out1,
    )
    print_scenario_result(r1)
    results.append(r1)

    # ── Scenario 2: Action video (frequent scene changes) ────────────
    out2 = str(Path(base_output) / "scenario_action")
    r2 = run_demo_scenario(
        name="Action video (frequent cuts)",
        description="Scene changes every 10 frames — tests cache miss handling.",
        n_frames=200,
        scene_change_every=10,
        similarity_threshold=float(os.getenv("WORLDCACHE_SIMILARITY_THRESHOLD", "0.92")),
        algorithm="dhash",
        output_dir=out2,
    )
    print_scenario_result(r2)
    results.append(r2)

    # ── Scenario 3: pHash algorithm ───────────────────────────────────
    out3 = str(Path(base_output) / "scenario_phash")
    r3 = run_demo_scenario(
        name="pHash algorithm",
        description="Perceptual (DCT-based) hash — more robust to lighting changes.",
        n_frames=150,
        scene_change_every=50,
        similarity_threshold=float(os.getenv("WORLDCACHE_SIMILARITY_THRESHOLD", "0.92")),
        algorithm="phash",
        output_dir=out3,
    )
    print_scenario_result(r3)
    results.append(r3)

    # ── Comparison table ──────────────────────────────────────────────
    print_comparison_table(results)

    # ── KV reuse schedule preview ─────────────────────────────────────
    demo_kv_reuse_schedule(out1)

    # ── Final summary ─────────────────────────────────────────────────
    best = max(results, key=lambda r: r["ram_reduction_pct"])
    cprint(f"\n[bold]Best RAM reduction:[/bold] [green bold]{best['ram_reduction_pct']:.1f}%[/green bold]")
    cprint(f"[dim]Output files written to: {base_output}/[/dim]")
    cprint(
        "\n[bold cyan]Next steps:[/bold cyan]\n"
        "  worldcache process  my_video.mp4 --output-dir out/\n"
        "  worldcache benchmark my_video.mp4\n"
        "  worldcache inspect  out/worldcache_stats.json\n"
    )
    cprint(
        "[dim]Built autonomously using NEO - your autonomous AI Agent https://heyneo.so[/dim]"
    )


if __name__ == "__main__":
    main()
