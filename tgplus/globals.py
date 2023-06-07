"""
Global definitions common to the data-loading, training code, app... 
"""
from pathlib import Path
import tgplus


# Root path of the project:
PROJECT_ROOT = Path(tgplus.__path__[0]).parent

# Path where we put any data loaded from the internet:
DATA_CACHE = PROJECT_ROOT / "data"


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
