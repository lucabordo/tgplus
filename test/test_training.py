from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from tgplus.globals import DATA_CACHE, GENRES_TAXONOMY, TextWithGenres
from tgplus.training import (
    Embedding,
    ScikitLearnPredictor,
    load_encoder,
    calculate_or_reload_embeddings,
    train_model
)


def get_dummy_text_with_genres() -> TextWithGenres:
    result = [
        (
            "A fun adventure with animated toys, from Disney",
            ['Animation', 'Comedy', 'Family']
        ),
        (
            "A child-friendly movie with adorable characters and good intentions",
            ['Animation', 'Comedy', 'Family']
        ),
        (
            "A tense psychological drame with lots of suspense; the story of a murder whose investigation reveals dark secrets",
            ["Thriller"]
        )
    ]

    # We have 32 classes and will have issues with some classifiers if they aren't all represented
    # somehow in the training data:
    for genre in GENRES_TAXONOMY:
        result.append((f"A movie with {genre} in it", [genre]))

    return result


@pytest.mark.parametrize("parallel", [False, True])
def test_basic_encoder(parallel):
    encoder_function = load_encoder(parallel=parallel)
    data = get_dummy_text_with_genres()
    embeddings = encoder_function([text for text, genres in data])
    assert len(embeddings) == len(data)
    assert all(isinstance(e, Embedding) for e in embeddings)
    assert all(e.ndim == 1 for e in embeddings)


def test_calculate_or_reload_embeddings():
    data = get_dummy_text_with_genres()
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "dummy.npy"
        # With allow_reuse, first call will exercise the code path that calculates and saves;
        # second call with exercise the code path that reloads the embeddings:
        embeddings1 = calculate_or_reload_embeddings(data, allow_reuse=True, path=path)
        embeddings2 = calculate_or_reload_embeddings(data, allow_reuse=True, path=path)
        # Data is the same initially calculated or reloaded:
        assert len(embeddings2) == len(embeddings1)
        for (text1, genres1, vector1), (text2, genres2, vector2) in zip(embeddings1, embeddings2):
            assert text1 == text2
            assert genres1 == genres2
            np.testing.assert_array_equal(vector1, vector2)


def test_train_model():
    """
    This exercices most of the model traing code and Predictor code:
    train a model on dummy data;
    save and reload;
    apply predictions to a dummy test set;
    check that the predictor gives the same prediction before and after being saved/reloaded.
    """
    data = get_dummy_text_with_genres()
    data_with_embeddings = calculate_or_reload_embeddings(data, allow_reuse=False, path=Path("dummy.npy"))
    predictor = train_model(data_with_embeddings)

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "dummy.joblib"
        predictor.save(path)
        new_predictor = ScikitLearnPredictor.load(path)
        genre1 = predictor("A fun adventure with animated toys, from Disney")
        genre2 = new_predictor("A fun adventure with animated toys, from Disney")
        assert genre1 == genre2