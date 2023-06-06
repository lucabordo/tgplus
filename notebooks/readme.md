Notebooks in this folder are for data exploration.

Notebooks in this folder can have some minimal testing, which verifies their imports. 
This is done by `test/test_notbooks.py`.

Notebooks are exempt of tests if they are put under a `notest` subfolder,
or are explicitly marked by a "-notest.ipynb" suffix.
Such notebooks should be expected to be very brittle and tend to break as the codebase evolves.
