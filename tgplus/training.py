"""
Code for training the model.
"""
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from tgplus.globals import Encoder, Embedding


def load_encoder(parallel: bool = True) -> Encoder:
    """
    Get a pre-trained encoder.
    """
    # The choice here is a bit arbitrary:
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def encode_function(text_data: Sequence[str]) -> Sequence[Embedding]:
        """
        Expose the model using the "Encoder" interface - independent
        of the encoder implemenration we use
        """
        if parallel:
            pool = encoder.start_multi_process_pool()
            results2d = encoder.encode_multi_process(text_data, pool=pool)
            assert isinstance(results2d, np.ndarray)
            assert results2d.ndim == 2
            SentenceTransformer.stop_multi_process_pool(pool)
            results: Sequence[Embedding] = tuple(results2d)
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
