"""Guardrail Router: two-stage screening for untrusted text."""

from .app import create_app

__version__ = "0.4.1"

__all__ = ["__version__", "create_app"]
