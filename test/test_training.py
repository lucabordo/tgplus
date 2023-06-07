from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from tgplus.globals import DATA_CACHE, TextWithGenres, Embedding
from tgplus.training import load_encoder, calculate_or_reload_embeddings


def get_dummy_text_with_genres() -> TextWithGenres:
    return [
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
