"""
Access and basic processing of movies data.
"""
import json
from pathlib import Path
from typing import List, Dict, TypeAlias, FrozenSet
import pandas as pd
import tgplus


# Root path of the project:
PROJECT_ROOT = Path(tgplus.__path__[0]).parent

# Path where we put any data loaded from the internet:
DATA_CACHE = PROJECT_ROOT / "data"


def get_movies_table() -> pd.DataFrame:
    """
    Load the full table of movies data, unfiltered and unprocessed.
    """
    assert DATA_CACHE.exists()
    movies_data_csv = DATA_CACHE / "movies_metadata.csv"
    if not movies_data_csv.exists():
        raise FileNotFoundError(f"CSV file needs to be manually populated under {movies_data_csv}")

    movies_table = pd.read_csv(movies_data_csv)
    # A different length here would mean a change of data version, let's not leave this silent:
    assert len(movies_table) == 45466
    return movies_table


# Dictionaries with keys "id" and "name":
Genre: TypeAlias = Dict


def parse_genre(genre_entry: str) -> List[Genre]:
    """
    Genres are all stored in these data in this JSON form:
    "[{'id': 12, 'name': 'Adventure'}, {'id': 14, 'name': 'Fantasy'},
      {'id': 10751, 'name': 'Family'}]"
    which needs tweaks to parse; always returning lists of dictionaries with "id" and "name".
    """
    curated = genre_entry.replace("'", '"')
    result = json.loads(curated)
    assert isinstance(result, list)
    assert all(isinstance(genre, dict) for genre in result)
    assert all(set(genre.keys()) == {"id", "name"} for genre in result)
    return result


def build_genre_mapping(movies_table: pd.DataFrame) -> FrozenSet[str]:
    """
    From the movies datable's "genre" column that has data like:
    "[{'id': 12, 'name': 'Adventure'}, {'id': 14, 'name': 'Fantasy'},
      {'id': 10751, 'name': 'Family'}]" ; 

    gather the set of all genre names used in the data.
    """
    return frozenset(
        genre["name"]
        for genre_list in movies_table["genres"]
        for genre in parse_genre(genre_list)
    )
