.PHONY: test
test:
	poetry run pytest tests --cov=neddf --cov-report term-missing --durations 5

.PHONY: format
format:
	poetry run pysen run format

.PHONY: lint
lint:
	poetry run pysen run lint