"""
Service package bootstrap.

Importing any module under ``services`` now automatically loads environment
variables from the closest ``.env`` so background workers (guardian,
production manager, wallet runner, Django) all inherit the same secrets even
when they start from different entrypoints.
"""

from .env_loader import EnvLoader

# Load exactly once per interpreter import. Safe to call repeatedly.
EnvLoader.load()

__all__ = ["EnvLoader"]
