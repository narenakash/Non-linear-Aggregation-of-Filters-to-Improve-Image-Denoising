import numpy as np

def getNeighbours(I, x, y, k):
    """
    load and reshape the training data 
    """
    assert(0 <= x - k and x + k < I.shape[0] and 0 <= y - k and y + k < I.shape[1])
    return I[x - k:x + k + 1, y - k:y + k + 1].flatten()