.PHONY: install lint test

install:
	pip install -r requirements.txt && pip install -e .

test:
	pytest -vv --cov=test tests

lint:
	pylint --disable=R,C src/vision_coder/
