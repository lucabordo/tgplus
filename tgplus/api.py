"""
API that exposes the movie prediction.

We use FastAPI, and haven't gone very far from the initial example in the doc
https://fastapi.tiangolo.com/
"""
from typing import List
from fastapi import FastAPI
from tgplus.training import ScikitLearnPredictor
from tgplus.globals import DATA_CACHE


# The app to be envoked by FastAPI:
app = FastAPI()


def get_predictor() -> ScikitLearnPredictor:
    """
    Load the predictor; 
    Note that we always reload fresh, which is terribly not optimal
    """
    if len(_PREDICTOR) == 0:
        model_path = DATA_CACHE / "model.joblib"
        if not model_path.exists():
            raise ValueError("Run the training.py script before running the evaluation.")
        predictor = ScikitLearnPredictor.load(model_path)
        assert isinstance(predictor, ScikitLearnPredictor)
        print("Model is loaded")
        _PREDICTOR.append(predictor)
    return _PREDICTOR[0]


# Global for use only by the load_predictor function ;
# For weird reasons I tried an Optional[ScikitLearnPredictor] defaulted to None but got errors,
# Mutating a list without using the global keyword to assign the module-level variable
# seems preferable:
_PREDICTOR: List[ScikitLearnPredictor] = []


@app.get("/{overview}")
def read_root(overview: str):
    """
    Entry point for the genre classification service.

    Note that this is not exactly as documented, 
    because I have note tuned GET / POST, the localhost, etc.
    """
    print("Received overview:", overview)
    predictor = get_predictor()
    predicted_genre = predictor(overview)
    print("Prediction:", predicted_genre)
    return {"genre": predicted_genre}


@app.get("/ping/{message}")
def ping(message: str):
    """
    Simpler entry point just to check that messages can be acknowledged.
    """
    return {"received": message}
