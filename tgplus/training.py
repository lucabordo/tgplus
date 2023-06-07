"""
Code for training the model.
"""
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from tgplus.data import get_movies_data
from tgplus.globals import (
    Encoder,
    Embedding, 
    DATA_CACHE, 
    TextWithGenres, 
    TextWithGenresAndEmbeddings
)


def load_encoder(parallel: bool = False) -> Encoder:
    """
    Get a pre-trained encoder.

    Note: the option for parallel code uses the parallelism backed by this model
    implementation - which without GPU would back to 
    """
    # The choice here is a bit arbitrary:
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def encode_function(text_data: Sequence[str]) -> Sequence[Embedding]:
        """
        Expose the model using as an "Encoder" function as defined by the Encoder type definition.
        This is independent of the encoder implementation and should allow us to switch and make an
        easy and principled comparison of implementations.
        """
        if parallel:
            # This parallel implementation works but doesn't pay off as much as hoped on my machine;
            # and it is frustrating not to have a progress bar:
            pool = encoder.start_multi_process_pool()
            results2d = encoder.encode_multi_process(text_data, pool=pool)
            results: Sequence[Embedding] = tuple(results2d)
            assert isinstance(results2d, np.ndarray)
            assert results2d.ndim == 2
            SentenceTransformer.stop_multi_process_pool(pool)
        else:
            results = tuple(
                encoder.encode(text)
                for text in tqdm(text_data, desc="Embedding")
            )

        for embedding in results:
            embedding.flags.writeable = False
            assert isinstance(embedding, np.ndarray)
            assert embedding.ndim == 1
            assert embedding.dtype == np.float32
        return results

    return encode_function


def calculate_or_reload_embeddings(
        data: TextWithGenres,
        allow_reuse: bool,
        path: Path,
) -> TextWithGenresAndEmbeddings:
    """
    Calculate and save, or reuse if previously calculated, a dataset enriched with embeddings.

    Note that embeddings, if saved, are just dumped as numpy arrays. This is hacky and violates
    a principle I'm keen on: self-containedness (which embedding corresponds to what data point!).
    Ideally and if serialization is really needed we'd evolve that to a proper way to save 
    TextWithGenresAndEmbeddings objects in a way that is self-contained, for instance as tabular
    data (saved, say, in parquet format).
    """
    if path.exists():
        print("Reloading embeddings from", path)
        embeddings2d = np.load(path)
        assert isinstance(embeddings2d, np.ndarray)
        assert embeddings2d.ndim == 2
        embeddings: Sequence[Embedding] = tuple(embeddings2d)
    else:
        print(f"Calculating embeddings for {len(data)} data points")
        encoder = load_encoder(parallel=False)
        embeddings = encoder([text for text, _genres in data])
        embeddings2d = np.array(embeddings)
        assert isinstance(embeddings2d, np.ndarray)
        assert embeddings2d.ndim == 2
        if allow_reuse:
            print("Saving embeddings to", path)
            np.save(path, embeddings2d)
    return [(text, genre, embedding) for (text, genre), embedding in zip(data, embeddings)]


def load_data_with_embeddings(
        test_fraction=0.1, 
        seed=1245737,
        allow_reuse=False
) -> Tuple[TextWithGenresAndEmbeddings, TextWithGenresAndEmbeddings]:
    """
    Load the train and test data and pre-calculate embeddings for all text.

    If the flag `allow_reuse` is set to True, we will cache the embeddings to the data folder
    for faster iteration, as calculating the embeddings for the training set is consuming. 

    This is a dangerous flag, hence False by default: cached embeddings may not correspond any more
    to any previous version of code or data. Saved npy arrays need to be cleaned by hand

    (this is just a hack to iterate fast given slow embedding code; ideally we'd use a proper
    pipeline or workflow for dealing with derived data, in a way that is aware of data and code 
    version)
    """
    training, test = get_movies_data(test_fraction=test_fraction, seed=seed)
    return (
        calculate_or_reload_embeddings(training, allow_reuse, 
                                       DATA_CACHE / "embeddings_training.npy"),
        calculate_or_reload_embeddings(test, allow_reuse,
                                        DATA_CACHE / "embeddings_test.npy")
    )


def main():
    """
    Entry point of the training script.
    """
    train, test = load_data_with_embeddings(allow_reuse=True)
    print(f"Done with embeddings - length: {len(train)}, {len(test)}")


if __name__ == "__main__":
    main()
