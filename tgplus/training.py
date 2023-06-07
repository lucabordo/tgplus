"""
Code for training the model.

This include a main that allows this script to run end to end and save a model.
The model can then be loaded by the service to be queried for predictions.

The approach we use here is:
- We use a pre-trained embedding model that encodes the text into a vector space
  of some pre-defined dimension. We do not fine-tune this model (yet the embedding
  computation consumes most of the runtime we could afford on a laptop)
- A simple model is then used that is trained from the features defined by the embdding
  of the input text, and with labels corresponding to a subset of the genres, that
  are one-hot encoded (this is multi-label, in the sense that a movie can be labelled
  as Adventure and as Family, for instance).

Note of the components selected for embedding and multilabel classification have been 
selected with enough care, reading about internals, tuning of hyper-parameters, etc.
A bit of care has been put into making the interplay between these components well-defined
so that in theory trying alternative encoders and/or classifiers should be easy;
ultimately we'd move towards making this configurable in order to apply some amount of 
model hyper-parameter tuning. This would need proper metrics though!
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple, TypeAlias, Callable, List

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from tgplus.data import get_movies_data
from tgplus.globals import (
    Predictor,
    TextWithGenres,
    DATA_CACHE,
    GENRES_TAXONOMY,
    one_hot_encode
)


# region Type definitions
# These types are custom for the specific models we are considering,
# and do not need to be exposed in globals.py which is for shared content

# A 1D numpy array of dtype float that represents text embedded in some vector space:
Embedding: TypeAlias = np.ndarray

# A (usually, pre-trained) model that encodes text into some embedding space;
# This takes inputs by batches to leave room for parallel (CPU or GPU) processing:
Encoder: TypeAlias = Callable[[Sequence[str]], Sequence[Embedding]]

# A TextWithGenres enriched with pre-calculated embeddings:
TextWithGenresAndEmbeddings: TypeAlias = List[Tuple[str, List[str], Embedding]]

# endregion 

# region Encoding (i.e. transformation of text into embeddings)

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

    return [
        (text, genre, embedding)
        for (text, genre), embedding in zip(data, embeddings)
    ]


def load_data_with_embeddings(allow_reuse=False) -> TextWithGenresAndEmbeddings:
    """
    Load the train data and pre-calculate embeddings for all text.

    If the flag `allow_reuse` is set to True, we will cache the embeddings to the data folder
    for faster iteration, as calculating the embeddings for the training set is consuming. 

    This is a dangerous flag, hence False by default: cached embeddings may not correspond any more
    to any previous version of code or data. Saved npy arrays need to be cleaned by hand

    (this is just a hack to iterate fast given slow embedding code; ideally we'd use a proper
    pipeline or workflow for dealing with derived data, in a way that is aware of data and code 
    version)
    """
    training, _test = get_movies_data()
    return calculate_or_reload_embeddings(
        training,
        allow_reuse,
        DATA_CACHE / "embeddings_training.npy"
    )

# endregion

# region Implementation of a predictor and of its training

@dataclass(frozen=True)
class ScikitLearnPredictor(Predictor):
    """
    A predictor based on a scikit learn model that is applies to embeddings
    that the text has to be converted to.
    """
    # The scikit-learn multi-label classifier used;
    # it may not need to be of type KNeighborsClassifier, we could probably generalise
    # the type - if there is an appropriate scikit-kearn type definition (too many mixins)
    # for classifiers that support multi-labelled data:
    classifier: KNeighborsClassifier

    # The encoder that the classifier has been trained against;
    # 
    encoder: Encoder = field(default_factory=load_encoder)

    # Override
    def __call__(self, desription: str) -> str:
        """
        Given a movie description, predict a genre.

        Note that, following the example in the doc, we return a single genre, even though
        the models will primarily be trained on data that has multiple genre labels per movie.
        """
        # Predictions should give us one probability per genre - in that order:
        prediction_size = len(GENRES_TAXONOMY)

        # Embed the text into vector space:
        [embedding] = self.encoder([desription])
        assert embedding.ndim == 1

        # Apply the classifier to the feature vector:
        predictions = [
            1.0 - proba.flatten()[0]
            for proba in self.classifier.predict_proba(embedding.reshape(1, -1))   
        ]

        # Get a (not necessarily unique) genre that is predicted with maximal probability:
        assert len(predictions) == prediction_size
        predicted_genre = max(
            range(prediction_size),
            key=lambda position: predictions[position]
        )

        # Return the genre that was picked by the one-hot encoding:
        return GENRES_TAXONOMY[predicted_genre]

    # Override
    def save(self, path: Path) -> None:
        """
        Save the model to disk.
        """
        # NOTE: joblib is sometimes recommended for serializing scikit-learn models
        #  though it seems to rely on pickling, which is brittle (perhaps not here)
        #  and which I usually prefer avoid - much prefer saving a model by saving
        # ala Torch just the weights into PT file. 
        joblib.dump(self.classifier, path)

    # Override
    @classmethod
    def load(cls, path: Path) -> "Predictor":
        """
        Load a saved model.
        """
        classifier = joblib.load(path)
        assert isinstance(classifier, KNeighborsClassifier)  # for now
        return ScikitLearnPredictor(classifier)


def train_model(training_data: TextWithGenresAndEmbeddings) -> Predictor:
    """
    Given a training dataset, create and train a model, giving a predictor.
    """
    # Get the embeddings and one-hot-encoded labels into the format 
    # expected by this classifier:
    training_x = np.array([
        embedding
        for _text, _genres, embedding in training_data
    ])
    training_y = np.array([
        one_hot_encode(genres)
        for _text, genres, _embedding in training_data
    ])

    # Train the model:
    classifier = KNeighborsClassifier()  # TODO: think the parameterization!
    classifier.fit(training_x, training_y)

    # Get a predictor:
    return ScikitLearnPredictor(classifier)

# endregion

# region scripting

def main():
    """
    Entry point of the training script.
    """
    # The goal of this script is to create the model in this path, for reuse by the API:
    model_path = DATA_CACHE / "model.joblib"

    training_data = load_data_with_embeddings(allow_reuse=False)
    print(f"Done generating embeddings - length: {len(training_data)}")

    predictor = train_model(training_data)
    print("Model is trained")

    predictor.save(model_path)
    print("Model saved under", model_path)


if __name__ == "__main__":
    main()

# endregion
