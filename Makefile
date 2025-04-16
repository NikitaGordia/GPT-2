#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = gpt-2
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
DOCKER_BUILDKIT=1

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Download and process fineweb dataset
.PHONY: fineweb
fineweb:
	python gpt/fineweb.py data/processed/fineweb_edu


## Train the model
.PHONY: train
train:
	torchrun --nproc_per_node=1 gpt/train.py


## Run tests
.PHONY: test
test:
	python -m pytest tests


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Build Docker image
.PHONY: docker-build
docker-build:
	docker compose build

## Run model training in Docker
.PHONY: docker-train
docker-train:
	docker compose up train

## Process data in Docker
.PHONY: docker-data
docker-data:
	docker compose up data

## Run tests in Docker
.PHONY: docker-test
docker-test:
	docker compose up base

## Debug Docker container with interactive shell
.PHONY: docker-debug
docker-debug:
	docker compose run --rm base /bin/bash


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
