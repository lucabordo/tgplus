import numpy as np
from tgplus.globals import one_hot_encode, one_hot_decode, GENRES_TAXONOMY


def test_one_hot_encoding():
    hot = one_hot_encode(["Action", "Adventure"])
    assert hot.shape == (len(GENRES_TAXONOMY), )
    # The following assert is brittle if we change taxonomy - but that's life, we'd fix it then!
    np.testing.assert_array_equal(
        hot, 
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    
    back = one_hot_decode(hot)
    assert back == ["Action", "Adventure"]
