"""
Global definitions common to the data-loading, training code, application.
"""
from pathlib import Path
from typing import List, Tuple, TypeAlias
import tgplus


# region Paths

# Root path of the project:
PROJECT_ROOT = Path(tgplus.__path__[0]).parent

# Path where we put any data loaded from the internet:
DATA_CACHE = PROJECT_ROOT / "data"

# endregion

# region Type definitions

# Some of these type definitions are simple aliases at the moment and could evolve
# to protocols or interfaces when/if needed. They serve as unique point of description
# for some data types, and allow to define clear function signatures for the data manipulation.

# Collection of text data annotated by genre; Each entry pairs: 
# - textual description of the movie;
# - list of genres associated to the movie.
# This is a common representation for data from any source, 
# and whether the collection is used for training, validation or test:
TextWithGenres: TypeAlias = List[Tuple[str, List[str]]]


class Predictor:
    """
    Abstract class for predictor; 
    these are the objects whose interface has to be agreed between modeling and deployment: 
    - training will save objects of this type; 
    - the API will load a predictor from disk, pass data to it, and expose its results.
    """

    def __call__(self, desription: str) -> str:
        """
        Given a movie description, predict a genre.

        Note that, following the example in the doc, we return a single genre, even though
        the models will primarily be trained on data that has multiple genre labels per movie.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "Predictor":
        """
        Load a saved model.
        """
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """
        Save the model to disk.
        """
        raise NotImplementedError

# endregion

# region Constants

# Immutable list of genres we use;
# This is sorted and order matters - indices will represent integer encodings:
GENRES_TAXONOMY = (
    'Action',
    'Adventure',
    'Animation',
    'Aniplex',
    'BROSTA TV',
    'Carousel Productions',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Family',
    'Fantasy',
    'Foreign',
    'GoHands',
    'History',
    'Horror',
    'Mardock Scramble Production Committee',
    'Music',
    'Mystery',
    'Odyssey Media',
    'Pulser Productions',
    'Rogue State',
    'Romance',
    'Science Fiction',
    'Sentai Filmworks',
    'TV Movie',
    'Telescene Film Group Productions',
    'The Cartel',
    'Thriller',
    'Vision View Entertainment',
    'War',
    'Western'
)

# endregion
