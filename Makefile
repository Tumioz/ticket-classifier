.PHONY: install train serve test lint format docker

install:
	pip install -e ".[dev]"

train:
	python -m ticket_classifier.train

serve:
	uvicorn ticket_classifier.api:app --reload

test:
	pytest --cov=ticket_classifier --cov-report=term-missing

lint:
	ruff check .

format:
	ruff check --fix . && ruff format .

docker:
	docker build -t ticket-classifier .
	docker run --rm -p 8000:8000 ticket-classifier
