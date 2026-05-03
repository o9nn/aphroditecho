"""Memory-membrane test conftest — does not require torch."""
import sys
from pathlib import Path

# Insert repo root so memory_boot import path works
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Block the parent conftest from running by collecting only this dir
collect_ignore_glob = ["*"]
