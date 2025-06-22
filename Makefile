.PHONY: help install install-dev test test-cov lint format clean build run docker-build docker-run publish

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install production dependencies"
	@echo "  install-dev - Install development dependencies"
	@echo "  test        - Run tests"
	@echo "  test-cov    - Run tests with coverage"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo "  clean       - Clean temporary files"
	@echo "  build       - Build the application"
	@echo "  run         - Run the application locally"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run Docker container"

# Installation
install:
	poetry install --without dev

install-dev:
	poetry install --with dev
	poetry run pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage .pytest_cache/

# Local development
build:
	poetry version patch
	poetry build

publish: build
	@echo "Publishing package to PyPI..."
	poetry version patch
	poetry publish

run:
	python -m uvicorn src.main:app --host 0.0.0.0 --port 8088 --reload

# Publishing
publish: build
	poetry publish

# Docker commands
docker-build:
	docker build -t european-invoice-ocr .

docker-run:
	docker-compose up -d

docker-dev:
	docker-compose -f docker-compose.dev.yml up

docker-stop:
	docker-compose down

# Setup environment
setup:
	chmod +x scripts/setup.sh
	./scripts/setup.sh

# Download models
download-models:
	python scripts/download_models.py

# Validate installation
validate:
	python scripts/validate_installation.py

# Run benchmarks
benchmark:
	python scripts/benchmark.py