FROM python:3.11-slim

WORKDIR /app

# Dependencies FIRST, keyed on pyproject.toml alone. The dependency layer is large (xgboost, pandas, scipy,
# scikit-learn) and used to sit AFTER `COPY src/`, so every commit rebuilt it with a new digest and the VPS kept a
# full extra copy per deploy. Now an unchanged dependency set reuses the SAME layer digest across commits and a deploy
# adds only the thin app layers.
# `pip install -e .` needs the package directory to exist to resolve it, so a stub stands in for the real source;
# the real source is copied right after and PYTHONPATH=/app/src (set below) is what the app imports from.
COPY pyproject.toml ./
RUN mkdir -p src/cfb_predictor && touch src/cfb_predictor/__init__.py \
    && pip install --no-cache-dir -e . \
    && rm -rf src

COPY src/ ./src/

# Create cache and tracking data directories explicitly
RUN mkdir -p /app/data/cache/games /app/data/cache/teams /app/data/cache/player_stats /app/data/cache/odds /app/models

COPY models/ /app/models/

# The precomputed games/predictions/player-props this deployment actually
# serves (see public_snapshot.py's module docstring) -- generated locally
# or by .github/workflows/refresh-public-snapshot.yml
# (`python -m cfb_predictor.public_snapshot`) and committed, not built in
# this image. Must exist before building.
COPY data/public_snapshot.json ./data/public_snapshot.json

# `data/cache/` is gitignored *and* .dockerignore'd, so every one of the
# mkdir'd cache directories above is empty in this image and stays empty on
# every cold start. The team-stats backfill that produced the 42,190-row
# reconciliation in data/team_stats.py filled a local checkout's copy; none of
# it is here. startup_seed.py makes a fresh container fill that gap itself,
# exactly once, and is a no-op from the second boot on (it reads the cache the
# fetcher would read, so "already seeded" and "the fetcher would re-request it"
# are the same statement). It is a module under src/ rather than a shell script
# so the decision is unit-tested rather than trusted, and so the existing
# `src/**` deploy trigger already covers it.
#
# `;` and not `&&`, and `exec` on the server: a seeding failure -- CFBD
# unreachable, quota exhausted -- must not stop the container from serving, and
# must not put it in a restart loop that re-attempts the bill on every restart.
# It costs one metered CFBD call per (season, week) in scope, 352 at the
# 2004-2025 default; narrow it with CFB_SEED_FROM_YEAR / CFB_SEED_TO_YEAR /
# CFB_SEED_WEEKS. With no CFBD_API_KEY it logs `outcome=blocked` and fetches
# nothing. See startup_seed.py's module docstring, and
# `python -m cfb_predictor.startup_seed --check` for a spend-nothing check.
ENV PYTHONPATH=/app/src
ENV PUBLIC_MODE=true

EXPOSE 8003
CMD ["sh", "-c", "python -m cfb_predictor.startup_seed; exec uvicorn cfb_predictor.api.main:app --host 0.0.0.0 --port ${PORT:-8003}"]