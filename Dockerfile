FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

COPY --from=frontend-build /app/frontend/dist ./frontend/dist

ENV PYTHONPATH=/app/src
ENV PUBLIC_MODE=true

EXPOSE 8003
CMD ["sh", "-c", "uvicorn cfb_predictor.api.main:app --host 0.0.0.0 --port ${PORT:-8003}"]
