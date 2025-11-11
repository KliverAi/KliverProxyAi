.PHONY: help install test test-cov test-watch run dev clean lint format

# Variables
PYTHON = poetry run python
PYTEST = poetry run pytest

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	poetry install

# Test commands
test:  ## Run all tests
	$(PYTEST)

test-cov:  ## Run tests with coverage report
	$(PYTEST) --cov=app --cov-report=html --cov-report=term-missing

test-watch:  ## Run tests in watch mode
	$(PYTEST) -f

test-unit:  ## Run only unit tests
	$(PYTEST) -m unit

test-integration:  ## Run only integration tests
	$(PYTEST) -m integration

test-verbose:  ## Run tests with verbose output
	$(PYTEST) -vv

test-failed:  ## Run only failed tests from last run
	$(PYTEST) --lf

# Run commands
run:  ## Run the application
	poetry run python run.py

dev:  ## Run the application in development mode with auto-reload
	poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Code quality
lint:  ## Run linter (ruff)
	poetry run ruff check app tests

format:  ## Format code with black
	poetry run black app tests

format-check:  ## Check code formatting without making changes
	poetry run black --check app tests

# Cleanup
clean:  ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml

clean-all: clean  ## Clean everything including virtual environment
	rm -rf .venv/

# Coverage
coverage:  ## Generate and open coverage report
	$(PYTEST) --cov=app --cov-report=html
	@echo "Opening coverage report..."
	open htmlcov/index.html || xdg-open htmlcov/index.html

# Docker (if needed in the future)
docker-build:  ## Build docker image
	docker build -t kliverai .

docker-run:  ## Run docker container
	docker run -p 8000:8000 kliverai

# Database/migrations (if needed in the future)
# migrate:  ## Run database migrations
# 	$(PYTHON) -m alembic upgrade head

# Git helpers
git-clean:  ## Remove all untracked files and directories
	git clean -fd

# Show current configuration
show-config:  ## Show current configuration
	@echo "Python version:"
	@$(PYTHON) --version
	@echo "\nInstalled packages:"
	@poetry show --tree

# Health check
health:  ## Check if the service is running
	curl -s http://localhost:8000/health | jq

# API documentation
docs:  ## Open API documentation in browser
	@echo "Opening API docs..."
	open http://localhost:8000/docs || xdg-open http://localhost:8000/docs
