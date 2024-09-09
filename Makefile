.DEFAULT_GOAL := help

.PHONY: run clean

VENV ?= .venv
# PYTHON = $(VENV)/bin/python3
# PIP = $(VENV)/bin/pip

PYTHON = $$(if [ -d $(PWD)/'$(VENV)' ]; then echo $(PWD)/"$(VENV)/bin/python3"; else echo "python3"; fi)
PIP = $(PYTHON) -m pip

setup: requirements.txt
	$(PIP) install -r requirements.txt

# run: $(VENV)/bin/activate
# 	$(PYTHON) app.py

# $(VENV)/bin/activate: requirements.txt
# 	python3 -m venv $(VENV)
# 	$(PIP) install -r requirements.txt

venv: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip setuptools wheel build
	$(PIP) install -U -r requirements.txt
	touch $(VENV)

clean:
	rm -rf __pycache__
	rm -rf $(VENV)

# make VENV=my_venv run
