"""Computed signals for a fixture, per spec `2026-10-01-fixture-signals-design.md`.

One module per signal. `trust` is the only one Phase 1 ships here, and only for
the moneyline — see that module's docstring for the measured counts behind it.
"""

from . import trust

__all__ = ["trust"]