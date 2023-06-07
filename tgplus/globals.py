"""
Global definitions common to the data-loading, training code, app... 
"""
from pathlib import Path
from typing import List, Tuple, TypeAlias, Callable, Sequence
import numpy as np
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

# A 1D numpy array of dtype float that represents text embedded in some vector space:
Embedding: TypeAlias = np.ndarray

# A (usually, pre-trained) model that encodes text into some embedding space;
# This takes inputs by batches to leave room for parallel (CPU or GPU) processing:
Encoder: TypeAlias = Callable[[Sequence[str]], Sequence[Embedding]]

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
