FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r dev && useradd -r -g dev trainer

WORKDIR /trainer

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

RUN chown trainer:dev /trainer
COPY --chown=trainer:dev . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

ENV PATH="/trainer/.venv/bin:$PATH"
RUN chmod +x ./entrypoints/trainer.sh

# After entrypoint runs on trainer user
ENTRYPOINT [ "./entrypoints/trainer.sh" ]
