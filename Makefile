.PHONY: black-check
black-check:
	poetry run black --check melon tests

.PHONY: black
black:
	poetry run black melon tests

.PHONY: flake8
flake8:
	poetry run flake8 melon

.PHONY: isort-check
isort-check:
	poetry run isort --check-only melon tests

.PHONY: isort
isort:
	poetry run isort melon tests

.PHONY: mypy
mypy:
	poetry run mypy melon

.PHONY: test
test:
	poetry run pytest tests --cov=melon --cov-report term-missing --durations 5

.PHONY: format
format:
	$(MAKE) black
	$(MAKE) isort

.PHONY: lint
lint:
	$(MAKE) black-check
	$(MAKE) isort-check
	$(MAKE) flake8
	$(MAKE) mypy

.PHONY: test-all
test-all:
	$(MAKE) black
	$(MAKE) flake8
	$(MAKE) isort
	$(MAKE) mypy
	$(MAKE) test
