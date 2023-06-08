# Goal

This is a little demo project exposing a simple service that 
predicts genre given a movie description.

# Cheat sheet

```bash
# Populate the training data - yes this has to be done manually:
cp some_original_download_folder/movies_data.csv ./data/

# Create the virtual environment and install any tooling needed for the project:
make dev

# Optionally, run a bunch of sanity checks on the code;
# this includes static and unit tests:
make checks

# Activate the virtual environment that has been populated:
source ./venv/bin/activate

# Run the training script, this could take about 25 minutes and
# should generate a data/model.joblib file:
python tgplus/training.py

# Optionally, run a model evaluation of the model, to get metrics;
# this evaluation is just sketched, at this stage:
python tgplus/evaluation.py

# Start the server running the API:
cd tgplus
uvicorn api:app
```

Note that the API could use some tuning, and can for now be queried as in the following examples:
```bash
curl "http://127.0.0.1:8000/A movie about penguins in Antarctica building a spaceship to go to Mars."

curl "http://127.0.0.1:8000/The true story of notorious gangster Romeo Corleone's downfall."
```
The first movie is classified as _Documentary_ :) , the second as _Crime_.

Our evaluation, very small scale, suggests an accuracy of 67% (with lots of caveats)
on a test split from the movies data - see the `eval.py` to see exactly what we mean by the test set (limited) and the metric reported.


# Assumptions

I've used a Windows machine - the project scaffolding is similar to things I've used
repeatedly in Mac and Linux, but no testing has been done for this one other than Windows.

I've used Python 3.10. There is no reason why versions 3.8ish + wouldn't work. 
But they haven't been tested, so I wouldn't bet.

No cloud resource or GPU was used for this small demo - this was purely written and ran on a 
laptop, on CPU, which means obiously very small-scale training.

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

# Structure of this repository

There are lots of files - perhaps an overkill - but many come from a template that I copy / adapt in my
data science projects and that provides I reuse a Makefile, tools for dealing with dependencies, configurations of 
static analysis tools that can be used optionally, etc. 

The `tgplus` folder contains all the code:
- `globals.py`: Some basic constant and type definitions, shared between the loading, training and api code.
- `data.py`: code for loading training and test data.
- `training.py`: all the ML code, and a main function that trains and saves a model used by the API.
- `evaluation.py`: embryonic code for model evaluation.
- `api.py`: the code for the API that loads a model saved by training and predicts movie genre using it.

You may find it interesting to look also at the `tests` folder, and the `notebooks` which is where I 
start most explorations (code is then largely migrated to the `tgplus` module). 

# Limitations

Note that this code implements a toy model - this is demo code only.

There is [todos](./todos.txt) file in which I kept notes while doing prototyping.
