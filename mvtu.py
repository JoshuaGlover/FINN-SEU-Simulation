import numpy as np

class MVTU(object):

    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def run(self, input):
        # XNOR + popcount
        activation = dot_xnor(self.weights, input)
        # Threshold
        activation = np.greater(activation, self.threshold)

        return activation[0][0].astype(int)

# Helper functions
'''
Performs dot product using XNOR through numpy broadcasting. Code referenced from:
https://stackoverflow.com/questions/19278313/numpy-matrix-multiplication-with-custom-dot-product
'''
def dot_xnor(a, b):
    out = np.sum(a[..., np.newaxis] == b[np.newaxis, ...], axis=1)
    return out
