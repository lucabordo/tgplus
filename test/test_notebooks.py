"""
Automated tests for the notebooks found, by convention, under the `notebooks` folder of the project.
Note that we cannot, in general, test notebooks end-to-end, as they tend to access data, or cloud
resources... What we do instead is test their *first two cells* (by convention: first is a description,
second has all imports). This helps verify that imports don't break, which is in practice the main 
painpoint where notebooks block refactoring and code evolution. 
"""
from pathlib import Path
from typing import Iterable, Optional

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

import tgplus


# When running locally, ignore folders like checkpoints that may contain extra notebooks:
IGNORED_FOLDERS = [
    ".ipynb_checkpoints",
    ".venv",
    "notest"
]


def get_project_root() -> Path:
    return Path(tgplus.__path__[0]).parent


def test_project_folder_structure():
    """
    Test that should make it easier to detect
    errors caused by renaming / folder structure.
    """
    root = get_project_root()
    assert (root / 'notebooks').exists()
    assert (root / 'test').exists()


def list_notebooks() -> Iterable[str]:
    """
    List all notebooks in the codebase,
    except those explicitly marked by a "-notest.ipynb" suffix
    or those whose path is somehow under a "notest" folder.
    """
    notebook_root = get_project_root()
    return (
        # For readability of the parametrized test we return a relative string,
        # rather than a directly usable path, so that running in pytest will 
        # display a short-ish path like test_all_notebooks[notebooks/playground.ipynb]
        str(f.relative_to(notebook_root))

        for f in notebook_root.glob("**/*.ipynb")
        if not (
            f.name.endswith('-notest.ipynb') or
            any(folder in str(f) for folder in IGNORED_FOLDERS)
        )
    )


def smoke_test_notebook(notebook_path: Path, cell_count: Optional[int] = None):
    """
    Execute a notebook - this is just for purposes of "smoke testing" aimed
    to check whether the notebook (partially) runs at all.

    If a cell_count is specified then only the corresponding amount of cells
    at the top of the notebook will be executed.
    """
    with notebook_path.open() as file_handle:
        notebook = nbformat.read(file_handle, as_version=4)

    # Mutate the notebook object so we just have the first cells:
    if cell_count is not None:
        notebook.cells = notebook.cells[:cell_count]

    proc = ExecutePreprocessor(
        timeout=600,
        kernel_name='tgplus'
    )
    proc.allow_errors = False
    proc.preprocess(notebook)


@pytest.mark.parametrize(
    'notebook_subpath', [book for book in list_notebooks()]
)
def test_all_notebooks(notebook_subpath):
    """
    Loop over all notebooks (in a separate test each thanks to parameterized),
    and run them superficially, up to the second cell.

    This is just aimed to test that the imports of the notebooks work OK,
    making such luxurious things as refactoring a tiny bit easier.
    """
    notebook_path = get_project_root() / notebook_subpath

    # Error here would mean we changed the testing iteration logic:
    assert notebook_path.exists()

    smoke_test_notebook(notebook_path, cell_count=2)
