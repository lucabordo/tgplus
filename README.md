# Goal

This is an exploratory python project.

# Setup

This project includes a Makefile that automates the steps for:
- The creation (reproducibly) of the virtual environment needed for working on the project: `make dev`;
- Any tooling on the code, such as static analysis, tests, to be run either locally or as part of a CI: `make checks`;
- Any other command frequently used in development/test/production and worth automating,
  for instance exposing a kernel for data science work within a notebooks, commands to update the venv, etc.

Note that the `make` command assumes that an appropriate version of Python (Python 3.10) is installed
as the executable `python`. This can be customised within the Makefile manually (for now) if the 
selected Python interpreter is present in another path. 

If there is a need to install the project within a container or environment that doesn't have `make`, 
one may still take inspiration from the (reasonably simple) commands used within the _Makefile_ - 
these may be run manually or in some other automated way. For instance the `make dev` command just 
invokes a simple series of calls to `pip`, that can also be typed manually or become part of a Dockerfile.
