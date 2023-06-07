.PHONY: docs
SHELL=/bin/bash

# Main Python command used to create the venv; in some cases we might use python instead:
PYTHON_COMMAND = python  # C:\Users\lucab\AppData\Local\Programs\Python\Python310\python.exe

# Name of the project:
PROJECT_NAME = tgplus

# Virtual environment folder:
VENV = .venv

# Name that is seen from Jupyter when exposing the kernel:
KERNEL_NAME = $(PROJECT_NAME)

# Detect the OS - this will be Windows_NT, Darwin or Linux:
ifeq ($(OS),Windows_NT)
	OS_NAME = $(OS)
else
	OS_NAME = $(shell uname -s)
endif

# Python command within the venv:
ifeq ($(OS_NAME),Windows_NT)
	PYTHON = $(VENV)\Scripts\python.exe
else
	PYTHON = $(VENV)/bin/python
endif


# Display what the Makefile sees as the OS name:
.PHONY: os
os:
	@echo $(OS_NAME)


# Create or update a virtual environment for the project - just the parts that can be deployed:
.PHONY: env
env:
	-$(PYTHON_COMMAND) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade -r requirements.txt -c constraints.txt
	$(PYTHON) -m pip install -e . -c constraints.txt


# Run type checking:
.PHONY: type
type:
	@echo mypy
	$(PYTHON) -m mypy --ignore-missing-imports $(PROJECT_NAME) || exit 1


# Run static code analysis / "linting":
.PHONY: lint
lint:
	@echo pylint
	$(PYTHON) -m pylint --rcfile=pylintrc $(PROJECT_NAME) || exit 1


# Run the tests on the core pieces of code;
# this is disjoint from the experimental code:
.PHONY: test
test:
	$(PYTHON) -m pytest -vs test || exit 1


# Run all static analysis / style compliance tools:
.PHONY: checks
checks: type lint test


# Install all packages specified in the constraints file.
# This should install a needlessly large superset of the dependencies needed for dev
# This command is interesting if one wants to check that pinned versions are stable:
# if we run `make stable` followed by `make lock`, do we see *any* diff in constraints.txt?
.PHONY: stable
stable: env
	$(PYTHON) -m pip install -r constraints.txt


# Snapshot the versions currently used in venv, without updating them;
# note that this assumes a venv properly populated (with latest "dev" requirements, usually);
# Best practice is to manually inspect and code reviews the changes to constraints.txt.
# Note also that the lock file (constraints.txt) will be different depending on which version
# of Python we generate from.
.PHONY: lock
lock:
	$(PYTHON) -m pip freeze --exclude-editable > constraints.txt


# Apply an auto-formatting tool that will automatically fix PEP8 compliance;
# NOTE that currently this is not exactly aligned with PEP8/style checks ("make style");
# Therefore this tool provide partial help, but reformatting will need to be manually inspected
# and `make style` after applying:
.PHONY: format
format:
	$(PYTHON) -m autopep8 --exit-code --recursive --in-place --aggressive --max-line-length 100 $(PROJECT_NAME)


# Upgrade the versions of all dependencies;
# This will update all dependencies to latest, which is likely to be a risky business;
# Currently we don't take into account the Mac dependencies, which are just for dev:
.PHONY: update
update:
	$(PYTHON_COMMAND) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade -r requirements.txt
	$(PYTHON) -m pip freeze --exclude-editable > constraints.txt


# Expose the virtual environment to jupyter;
# This allows to do do some exploratory data programming within notebooks that use the 
# virutal environment specified for the project:
.PHONY: kernel
kernel:
	$(PYTHON) -m ipykernel install --user --name=$(KERNEL_NAME)


# Clean the environment and kernel, etc:
.PHONY: clean
clean:
	-$(PYTHON) -m jupyter kernelspec uninstall -y $(KERNEL_NAME)
	-rm -r $(VENV)


# Initialize a dev environment - virtual env, extra stuff needed for testing and dev:
.PHONY: dev kernel
dev: env
	$(PYTHON) -m pip install --upgrade -r requirements-dev.txt -c constraints.txt
