# shell used by make
SHELL=/bin/bash

# global variables
SRC_NAME=pytorch_project_template
SCRIPTS_NAME=scripts
VENV_NAME=venv
PYTHON=$(VENV_NAME)/bin/python3
EGG=$(SRC_NAME).egg-info

all: setup format-and-lint
	echo Setup complete

format-and-lint: 
	$(PYTHON) -m isort --profile black $(SRC_NAME)
	$(PYTHON) -m isort --profile black $(SCRIPTS)
	$(PYTHON) -m black --line-length 120 $(SRC_NAME)
	$(PYTHON) -m black --line-length 120 $(SCRIPTS)
	$(PYTHON) -m pytype --jobs auto --keep-going $(SRC_NAME)
	$(PYTHON) -m pytype --jobs auto --keep-going $(SCRIPTS_NAME)
	$(PYTHON) -m flake8 --ignore=E203,W503 --max-line-length 120 --statistics $(SRC_NAME)
	$(PYTHON) -m flake8 --ignore=E203,W503 --max-line-length 120 --statistics $(SCRIPTS)	

# Project setup
.PHONY: setup
setup: $(EGG)
$(EGG): $(PYTHON) setup.py requirements.txt
	$(PYTHON) -m pip install -U pip setuptools==59.1.1 wheel
	$(PYTHON) -m pip install -Ue .

$(PYTHON):
	python3.8 -m pip install virtualenv
	python3.8 -m virtualenv $(VENV_NAME)

# Careful!
# useful e.g. after modifying __init__.py files
destroy-setup: confirm-destroy
	rm -rf $(EGG)
	rm -rf $(VENV_NAME)

confirm-destroy:
	@( read -p "Destroy env? [y/n]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )
