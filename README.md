# GPT-2

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/"><img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" /></a> <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" /> <img src="https://img.shields.io/badge/Docker-Supported-blue?logo=docker" /> <img src="https://img.shields.io/badge/uv-Package%20Manager-blue?logo=python" /> <img src="https://img.shields.io/badge/PyTorch-Supported-orange?logo=pytorch" /> <img src="https://img.shields.io/badge/GPU-NVIDIA-green?logo=nvidia" /> <img src="https://img.shields.io/badge/DVC-Data%20Version%20Control-945DD6?logo=dvc" />

Production ready GPT-2

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── .cache         <- Cache for hellaswag or any other datasets.
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── entrypoints        <- Shell scripts for Docker entrypoints
│   └── trainer.sh     <- Entrypoint script for the training container
│
├── logs               <- Training logs and TensorBoard event files
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         gpt and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── tests              <- Test suite for the project
│   └── test_data.py   <- Tests for data processing functionality
│
├── docker-compose.yml <- Docker Compose configuration for containerized execution
│
├── trainer.Dockerfile <- Dockerfile for the training container
│
├── uv.lock            <- Lock file for uv package manager
│
└── gpt                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes gpt a Python module
    │
    ├── data.py        <- Data loading and processing utilities
    │
    ├── fineweb.py     <- Fineweb dataset handling
    │
    ├── hellaswag.py   <- HellaSwag dataset handling
    │
    ├── learning_rate.py <- Learning rate scheduling utilities
    │
    ├── model.py       <- GPT-2 model definition and configuration
    │
    ├── train.py       <- Training loop and related functionality
    │
    └── utils.py       <- Utility functions used across the project
```

--------

## Development Setup

### Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager

### Setting Up the Development Environment

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd gpt-2
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```
   This will ensure code quality checks run automatically before each commit.

   The pre-commit hooks will:
   - Run `ruff` to lint your code and fix common issues
   - Run `ruff-format` to format your code consistently

   You can manually run the hooks on all files with:
   ```bash
   pre-commit run --all-files
   ```

### Common Development Commands

- **Download and process Fineweb dataset (train/val)**:
  ```bash
  make fineweb
  ```

- **Run training**:
  ```bash
  make train
  ```

- **Run tests**:
  ```bash
  make test
  ```

## Docker Usage

This project includes Docker support for easy setup and reproducible environments.

### Prerequisites

- Docker and Docker Compose installed on your system
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

### Building the Docker image

```bash
docker compose build
```

### Available Services

1. **Training the model**:
   ```bash
   docker compose up train
   ```

2. **Processing data**:
   ```bash
   docker compose up data
   ```

3. **Running tests**:
   ```bash
   make docker-test
   ```

3. **Running an interactive shell for debugging**:
   ```bash
   make docker-debug
   ```

### Custom commands

You can run any command in the Docker container:

```bash
docker compose run --rm base python -m gpt.your_script
```

