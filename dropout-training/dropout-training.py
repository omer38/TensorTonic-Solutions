import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x, dtype=float)
    if rng is not None:
        rand = rng.random(x.shape)
    else:
        rand = np.random.random(x.shape)
    dropout_pattern = np.where(rand < (1 - p), 1.0 / (1 - p), 0.0)
    output = x * dropout_pattern
    return (output, dropout_pattern)