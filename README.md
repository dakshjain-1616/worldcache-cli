# WorldCache CLI – Pre-process video for world models at the edge

> *Made autonomously using [NEO](https://heyneo.so) · [![Install NEO Extension](https://img.shields.io/badge/VS%20Code-Install%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-82%20passed-brightgreen.svg)]()

> Deduplicate video frames before world model inference — cuts redundant processing by 70-90% using perceptual hashing

## Install

```bash
git clone https://github.com//worldcache-cli
cd worldcache-cli
pip install -r requirements.txt
```

## Quickstart

```bash
# Process video with deduplication
worldcache process input.mp4 --threshold 0.95 --output-dir frames/

# View cache statistics
worldcache stats --cache-dir frames/
```

## Key features

- Perceptual hashing for visual similarity detection
- Configurable deduplication threshold (0.0-1.0)
- Frame metadata preservation
- CLI and Python API support
- Zero GPU dependencies

## Run tests

```bash
pytest tests/ -q
# 82 passed
```

## Project structure

```
worldcache/
├── cache.py       # Frame deduplication logic
├── hasher.py      # Perceptual hashing
├── processor.py   # Video pipeline
└── cli.py         # Command interface
```