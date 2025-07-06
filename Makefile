.PHONY: install install-cpu lint test

install:
	pip install -r requirements.txt && pip install -e .

install-cpu:
	pip install -r requirements-cpu.txt && pip install -e .

test:
	pytest -vv tests

lint:
	pylint --disable=R,C src/vision_coder/
