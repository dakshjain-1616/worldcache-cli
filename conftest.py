"""
Root conftest.py — anchors pytest rootdir to the workspace and ensures
the `worldcache` package is importable without a prior `pip install`.
"""
import sys
import os

# Get the directory containing this conftest.py file
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = current_dir

# Set environment variable to indicate we're running tests
os.environ['WORLDCACHE_TESTING'] = '1'

# Make sure the workspace root is on sys.path so `import worldcache` works
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Also add the parent directory in case workspace is a subdirectory
parent_dir = os.path.dirname(workspace_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def pytest_configure(config):
    """Configure pytest."""
    pass

def pytest_sessionstart(session):
    """Set up session-level configurations."""
    pass

def pytest_collection_modifyitems(items):
    """Ensure tests are collected from the workspace root."""
    pass

def pytest_runtest_setup(item):
    """Additional setup before each test run."""
    pass
