Notebooks in this folder are for data exploration.
They allow to quickly take a look at the data, import some functions from the codebase, 
and play around.

Notebooks in this folder have some minimal testing, which verifies their imports. 
This is done by `test/test_notbooks.py`.

Notebooks are exempt of tests if they are put under a `notest` subfolder,
or are explicitly marked by a "-notest.ipynb" suffix.
Such notebooks should be expected to be very brittle and tend to break as the codebase evolves.
