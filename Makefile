.PHONY: black-check
black-check:
	poetry run black --check neddf tests

.PHONY: black
black:
	poetry run black neddf tests

.PHONY: flake8
flake8:
	poetry run flake8 neddf

.PHONY: isort-check
isort-check:
	poetry run isort --check-only neddf tests

.PHONY: isort
isort:
	poetry run isort neddf tests

.PHONY: mypy
mypy:
	poetry run mypy neddf

.PHONY: test
test:
	poetry run pytest tests --cov=neddf --cov-report term-missing --durations 5

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
