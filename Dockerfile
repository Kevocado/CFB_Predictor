FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Create cache and tracking data directories explicitly
RUN mkdir -p /app/data/cache/games /app/data/cache/teams /app/data/cache/player_stats /app/data/cache/odds /app/models

COPY models/ /app/models/

# The precomputed games/predictions/player-props this deployment actually
# serves (see public_snapshot.py's module docstring) -- generated locally
# or by .github/workflows/refresh-public-snapshot.yml
# (`python -m cfb_predictor.public_snapshot`) and committed, not built in
# this image. Must exist before building.
COPY data/public_snapshot.json ./data/public_snapshot.json

ENV PYTHONPATH=/app/src
ENV PUBLIC_MODE=true

EXPOSE 8003
CMD ["sh", "-c", "uvicorn cfb_predictor.api.main:app --host 0.0.0.0 --port ${PORT:-8003}"]