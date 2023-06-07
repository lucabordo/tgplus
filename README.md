# Goal

This is an exploratory python project.

# Quick summary

```bash
# Populate the training data 
cp some_original_download_folder/movies_data.csv ./data/

# Create the virtual environment and tooling for the project:
make dev

# Optionally, run a bunch of sanity checks on the code;
# this includes static and unit tests:
make checks

# Activate the virtual environment that has been populated:
source ./venv/bin/activate

# Run the training script, this could take 25 minutes and
# should generate a data/model.joblib file:
python tgplus/training.py

# Run the api
python tgplus/api.py
```

# Assumptions

I've used a Windows machine - the project scaffolding is similar to things I've used
repeatedly in Mac and Linux, but no testing has been done for this one other than Windows.

I've used Python 3.10. There is no reason why versions 3.8ish + wouldn't work. 
But they haven't been tested, so I wouldn't bet.

# Setup

This project includes a Makefile that automates the steps for:
- The creation (reproducibly) of the virtual environment needed for working on the project: `make dev`;
- Any tooling on the code, such as static analysis, tests, to be run either locally or as part of a CI (note done): `make checks`;
- Any other command frequently used in development/test/production and worth automating,
  for instance exposing a kernel for data science work within a notebooks, commands to update the venv, etc.

Note that the `make` command assumes that an appropriate version of Python (I only tested Python 3.10) is installed
as the executable `python`. This can be customised within the Makefile manually (for now) if the 
selected Python interpreter is present in another path. 

If there is a need to install the project within a container or environment that doesn't have `make`, 
one may still take inspiration from the (reasonably simple) commands used within the _Makefile_ - 
these may be run manually or in some other automated way. For instance the `make dev` command just 
invokes a simple series of calls to `pip`, that can also be typed manually or become part of a Dockerfile.
