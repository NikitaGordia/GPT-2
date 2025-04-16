FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS uv_builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r app && useradd -r -g app app

RUN mkdir -p /app && chown app:app /app

WORKDIR /app

COPY --from=uv_builder --chown=app:app /app /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

USER app

# Make dvc run without copying .git folder
RUN dvc config core.no_scm true
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

CMD ["torchrun", "--nproc_per_node=1", "gpt/train.py"]
