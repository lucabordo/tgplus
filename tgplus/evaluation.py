"""
Model evaluation;

This is very basic ATM, just suggest how we can make an evaluation that
is independent of the actual implementation details of the model, allowing to 
switch models and parameters and iterate towards good performance.

Needless to say this part would normally become all-important and should be
fleshed out as early as possible in the project - this drives iterations on the model.
"""
from typing import Sequence, List, Dict
from tqdm import tqdm

from tgplus.data import get_movies_data
from tgplus.training import ScikitLearnPredictor
from tgplus.globals import DATA_CACHE


def calculate_metric(
        predictions: Sequence[str],
        ground_truth: Sequence[List[str]],
        verbose: bool = True
) -> Dict[str, float]:
    """
    Example of metric - in simple cases it is simply about
    comparing predictions versus ground truth.
    """
    # Note that the ground truth has multiple genres for a movie
    # while our predictor currently predicts a single one;
    # This isn't ideal, we should probably allow multiple genres
    # to be predicted so that we can measure using F1 or any score
    # that reflects model quality as best as we can;
    # in this case it's the api that would somehow expose only a single
    # genre, if that is the requirement as suggested in the doc.

    assert len(predictions) == len(ground_truth)

    if verbose:
        print("First 10 examples of predictions versus ground truth")
        for predicted_genre, true_genres in zip(predictions[:10], ground_truth[:10]):
            print("    ", predicted_genre, true_genres)

    # For now we comppute the percentage of cases in which the single
    # returned genre is indeed within the correct ones given by the labels:

    count_correct = sum(
        predicted_genre in true_genres
        for predicted_genre, true_genres in zip(predictions, ground_truth)
    )
    return {"PERCENTAGE_HITS": count_correct / len(predictions)}


def main() -> None:
    """
    Entry point of the training script.
    """
    # Load a predictor:
    model_path = DATA_CACHE / "model.joblib"
    if not model_path.exists():
        raise ValueError("Run the training.py script before running the evaluation.")
    predictor = ScikitLearnPredictor.load(model_path)
    print("Model is loaded", predictor)
    
    # It is important here to load test data configured consistently with the training
    # code so that we have indeed disjoint data (same split ratio and seed - defaults):
    _training, test_data = get_movies_data()
    print(f"Done generating embeddings - length: {len(test_data)}")

    # HACK to make this fast to run, as embeddings will be generated:
    # print("WARNING - the data is throttled to keep this demo short!!")
    # test_data = test_data[:500]

    # Ground truth is easy to extract:
    ground_truth = [genres for _description, genres in test_data]

    # Make predictions - this will run embeddings:
    predictions = [
        predictor(description)
        for description, _genres in tqdm(test_data, desc="predictions")
    ]

    report = calculate_metric(predictions, ground_truth)
    print("METRIC:", report)


if __name__ == "__main__":
    main()
