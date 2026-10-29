# EchoJEV web app: backend + frontend + the trained artifacts in artifacts/. CPU-only PyTorch.
# Not in the image: data/, runs/, .env, and the Hugging Face models (WavLM, LAYA), which come from the
# host's Hugging Face cache mounted at /hf_cache (compose.yaml). Build and run: docker compose up --build

# --- build: the locked dependencies in a virtualenv (uv stays in this stage) ---
FROM python:3.11-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv
ENV VIRTUAL_ENV=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
# uv.lock versions, minus what the app never imports (training-only datasets, matplotlib, torchaudio)
# and the CUDA wheels the lock pins on Linux; PyTorch comes from the CPU index instead.
RUN uv venv /opt/venv \
    && uv export --frozen --no-dev --no-emit-project --no-hashes --no-header \
        --prune datasets --prune matplotlib --prune torchaudio -o /tmp/locked.txt \
    && grep -v -E '^(torch|triton|nvidia-|cuda-)' /tmp/locked.txt > /tmp/requirements.txt \
    && uv pip install --no-cache --index-url https://download.pytorch.org/whl/cpu \
        "torch==$(grep -oP '^torch==\K[0-9.]+' /tmp/locked.txt | head -1)" \
    && uv pip install --no-cache -r /tmp/requirements.txt \
    && find /opt/venv -depth \( -name __pycache__ -o -name tests \) -type d -exec rm -rf {} +

# Only the newest complete artifact per system (what the app serves), its threshold.json and
# rankings.json; older trainings stay on the build machine.
COPY artifacts ./artifacts
RUN python - <<'EOF'
import shutil
from pathlib import Path
for system in (p for p in Path("artifacts").iterdir() if p.is_dir()):
    runs = sorted(p for p in system.iterdir() if p.is_dir())
    complete = [p for p in runs if (p / "COMPLETE").is_file()]
    for old in (p for p in runs if p not in complete[-1:]):
        shutil.rmtree(old)
EOF

# --- runtime: Python, the virtualenv and the app ---
FROM python:3.11-slim
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY configs ./configs
COPY src ./src
COPY frontend ./frontend
COPY --from=build /app/artifacts ./artifacts

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf_cache \
    ADF_SECRETS_DIR=/secrets \
    ADF_SYSTEM=d2
LABEL org.opencontainers.image.source="https://github.com/Qwenty228/GC2-EchoJEV" \
      org.opencontainers.image.description="EchoJEV audio deepfake detection web app (CPU)"
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
