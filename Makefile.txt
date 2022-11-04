build-client:
	cd frontend && npm install && PUBLIC_DEV_MODE=PROD npm run build && rm -rf ../static && cp -r build/ ../static/
build-dev:
	cd frontend && npm install && PUBLIC_DEV_MODE=DEV npm run build-dev && rm -rf ../static && cp -r build/ ../static/
run-front-dev:
	cd frontend && npm install && PUBLIC_DEV_MODE=DEV npm run dev
run-dev:
	rm -rf .data/ && FLASK_DEBUG=development python app.py
run-prod:
	python app.py
build-all: run-prod

.PHONY: quality style test


check_dirs := tests src utils setup.py


quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)
	mypy src
	python utils/check_static_imports.py

style:
	black $(check_dirs)
	isort $(check_dirs)
	python utils/check_static_imports.py --update-file

test:
	pytest ./tests/