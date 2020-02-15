from __future__ import division
import pickle
import random
import numpy as np
import sys

class FINN(object):

    def __init__(self, sizes):
        # Network size attributes
        self.num_layers = len(sizes)
        self.layer_sizes = sizes

        # MVTU node masks
        self.inversion_mask  = [np.zeros((size,1)) for size in sizes]
        self.stuck_high_mask = [np.zeros((size,1)) for size in sizes]
        self.stuck_low_mask  = [np.zeros((size,1)) for size in sizes]

        file_suffix = ("_" + str(sizes[1])) * 2 + ".txt"

        # Load pretrained weights
        weight_file = "weights/weights" + file_suffix
        with open(weight_file, "rb") as fp:
            self.weights = pickle.load(fp)

        # Load pretrained thresholds
        threshold_file = "thresholds/thresholds" + file_suffix
        with open(threshold_file, "rb") as fp:
            self.thresholds = pickle.load(fp)

    def feedforward(self, a):
        # Binarise inputs (round to closest out of 0 or 1)
        a = np.around(a)

        # Perform XNOR dot product and thresholding on all layers
        for layer in range(self.num_layers - 1):
            # XNOR dot product and popcount
            a = dot_xnor(self.weights[layer], a)
            # Threshold
            a = np.greater_equal(a, self.thresholds[layer])

        return a

    def feedforward_node_seu(self, a, inversion=False, stuck_high=False, stuck_low=False):
        # Binarise inputs (round to closest out of 0 or 1)
        a = np.around(a)
        if inversion:
            a = invert_node(a, self.inversion_mask[0])
        elif stuck_high:
            a = raise_node(a, self.stuck_high_mask[0])
        elif stuck_low:
            a = lower_node(a, self.stuck_low_mask[0])

        # Perform XNOR dot product and thresholding on all layers
        for layer in range(self.num_layers - 1):
            # XNOR dot product and popcount
            a = dot_xnor(self.weights[layer], a)
            # Threshold
            a = np.greater_equal(a, self.thresholds[layer])
            if inversion:
                a = invert_node(a, self.inversion_mask[layer + 1])
            elif stuck_high:
                a = raise_node(a, self.stuck_high_mask[layer + 1])
            elif stuck_low:
                a = lower_node(a, self.stuck_low_mask[layer + 1])

        return a

    def evaluate(self, test_data, inversion=False, stuck_high=False, stuck_low=False):
        # Form tuples of prediction and target class for all test samples
        if inversion or stuck_high or stuck_low:
            test_results = [(np.argmax(self.feedforward_node_seu(x, inversion=inversion, \
                stuck_high=stuck_high, stuck_low=stuck_low)), y) for (x, y) in test_data]
        else:
            test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        # Return number of predictions which match target class
        return sum(int(x == y) for (x, y) in test_results)

    def weight_seu_at(self, i, j, k):
        self.weights[i][j][k] = bitflipZero(self.weights[i][j][k])

    def weight_seu(self):
        (i, j, k) = self.random_weight_index()
        self.weights[i][j][k] = bitflipZero(self.weights[i][j][k])
        print"Weight Flip at [{}][{}][{}]".format(i, j, k)
        return "\t {} \t {} \t {}".format(i, j, k)

    def node_inversion_at(self, layer, node):
        self.inversion_mask[layer][node] = 1
        print"Node Inversion at [{}][{}]".format(layer, node)

    def node_stuck_high_at(self, layer, node):
        self.stuck_high_mask[layer][node] = 1
        print"Node Stuck High at [{}][{}]".format(layer, node)

    def node_stuck_low_at(self, layer, node):
        self.stuck_low_mask[layer][node] = 1
        print"Node Stuck Low at [{}][{}]".format(layer, node)

    # Method that returns the index of a random weight or edge of the network
    def random_weight_index(self):
        # Find number of weights between each layer
        w_vec_lens = [i*j for (i,j) in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        num_weights = sum(w_vec_lens)

        # Choose a layer
        index = random.randint(1, num_weights)
        index_i = 0
        for (i,val) in enumerate(w_vec_lens):
            if index <= val:
                index_i = i
                break
            else:
                index -= val

        # Choose second and third indexes
        index_j = random.randint(0, self.layer_sizes[index_i+1]-1)
        index_k = random.randint(0, self.layer_sizes[index_i]-1)

        return (index_i, index_j, index_k)

# Helper functions
'''
Performs dot product using XNOR through numpy broadcasting. Code referenced from:
https://stackoverflow.com/questions/19278313/numpy-matrix-multiplication-with-custom-dot-product
'''
def dot_xnor(a, b):
    out = np.sum(a[..., np.newaxis] == b[np.newaxis, ...], axis=1)
    return out

def bitflipZero(a):
    return 1 if a == 0 else 0

def invert_node(a, mask):
    not_a = np.logical_not(a)
    not_mask = np.logical_not(mask)
    a = np.logical_or(np.logical_and(a, not_mask), np.logical_and(not_a, mask))
    return a

def raise_node(a, mask):
    not_mask = np.logical_not(mask)
    a = np.logical_and(a, not_mask) + mask
    return a

def lower_node(a, mask):
    not_mask = np.logical_not(mask)
    a = np.logical_and(a, not_mask)
    return a
