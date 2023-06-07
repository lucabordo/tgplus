"""
Access and basic processing of movies data.
"""
import json
from typing import List, Tuple

import pandas as pd
from numpy.random import RandomState

from tgplus.globals import GENRES_TAXONOMY, DATA_CACHE, TextWithGenres


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


def get_movies_data(test_fraction=0.1) -> Tuple[TextWithGenres, TextWithGenres]:
    """
    Load a dataset that has a bit of pre-processing and some splitting.

    For now we just use a training and test set - shamefully not validation
    (I see it unlikely I'd use it within time frame).
    """
    # Load as data frame:
    movies_table = get_movies_table()

    # There is some non-English content and we focus on English, for now:
    english_subset = movies_table[movies_table["original_language"] == "en"]

    # We gather text (we use overview but also - debatable - title) and genres for each entry:
    data = []
    subcolumns = zip(english_subset["title"], english_subset["overview"], english_subset["genres"])

    for title, overview, genres in subcolumns:
        text = f"{title}. {overview}"
        genre_names = parse_genres(genres)
        data.append((text, genre_names))
    
    # Split deterministically into two collections:
    prng = RandomState(1245737)
    test_size = int(len(data) * test_fraction)
    test_indices = set(prng.choice(len(data), test_size, replace=False))
    
    test_subset = [entry for index, entry in enumerate(data) if index in test_indices]
    train_subset = [entry for index, entry in enumerate(data) if index not in test_indices]

    return train_subset, test_subset
