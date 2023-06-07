"""
Model evaluation;

I won't go very far here, just suggest how we can make an evaluation that
is independent of the actual implementation details of the model, allowing to 
switch models and parameters and iterate towards good performance.

Needless to say this part would normally become all-important and should be
fleshed out as early as possible in the project - this drives iterations on the
model.
"""
from tgplus.data import get_movies_data
from tgplus.training import ScikitLearnPredictor
from tgplus.globals import DATA_CACHE


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
    _training, test = get_movies_data()
    print(f"Done generating embeddings - length: {len(test)}")


if __name__ == "__main__":
    main()
