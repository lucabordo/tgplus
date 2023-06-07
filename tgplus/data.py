"""
Access and basic processing of movies data.
"""
import json
from typing import List
import pandas as pd
from tgplus.globals import GENRES_TAXONOMY, DATA_CACHE


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
    
    # The list of genre names has been pre-computed;
    # let's check that any data loaded matches our taxonomy
    genre_names_in_data = {
        name
        for genres in movies_table["genres"]
        for name in parse_genres(genres)
    }
    assert genre_names_in_data.issubset(GENRES_TAXONOMY)

    return movies_table


def parse_genres(genre_entry: str) -> List[str]:
    """
    Get the list of genre names from an entry in the data frame. 

    Genres are all stored in these data in this JSON form:
    "[{'id': 12, 'name': 'Adventure'}, {'id': 14, 'name': 'Fantasy'},
      {'id': 10751, 'name': 'Family'}]"
    which needs:
    - tweaks to be parsed (not proper JSON simply because of quote style?)
    - extraction of the "name" - we ignore the ID - we'll use our own, contiguous integers.
    """
    curated = genre_entry.replace("'", '"')
    result = json.loads(curated)
    assert isinstance(result, list)
    assert all(isinstance(genre, dict) for genre in result)
    assert all(set(genre.keys()) == {"id", "name"} for genre in result)
    return [genre["name"] for genre in result]
