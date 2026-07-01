.PHONY: train serve test lint docker-build docker-up docker-down

train:
	set PYTHONPATH=. && python src/train.py

serve:
	set PYTHONPATH=. && uvicorn api.main:app --reload --port 8000

test:
	set PYTHONPATH=. && pytest tests/ -v

lint:
	ruff check src/ api/ tests/

docker-build:
	docker build -t ml-api:latest .

docker-up:
	docker compose up --build

docker-down:
	docker compose down
