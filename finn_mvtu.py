from __future__ import division
import mvtu
import pickle
import numpy as np

class FINN(object):

    def __init__(self, sizes):
        # Network size attributes
        self.num_layers = len(sizes)
        self.layer_sizes = sizes

        file_suffix = ("_" + str(sizes[1])) * 2 + ".txt"

        # Load pretrained weights
        weight_file = "weights/weights" + file_suffix
        with open(weight_file, "rb") as fp:
            self.weights = pickle.load(fp)

        # Load pretrained thresholds
        threshold_file = "thresholds/thresholds" + file_suffix
        with open(threshold_file, "rb") as fp:
            self.thresholds = pickle.load(fp)

        self.mvtu_network = []

        # Create MVTU for each node (excluding input layer)
        for layer in range(self.num_layers - 1):
            # print("Layer: " + str(layer))
            mvtu_layer = []
            for node in range(self.layer_sizes[layer+1]):
                weights = self.weights[layer][node,:]
                threshold = self.thresholds[layer][node]
                # print("Weights: " + str(weights) + " Threshold: " + str(threshold))
                mvtu_layer.append(mvtu.MVTU(weights, threshold))

            self.mvtu_network.append(mvtu_layer)

    def feedforward(self, a):
        # Binarise inputs (round to closest out of 0 or 1)
        in_act = np.around(a)

        # Perform XNOR dot product and thresholding on all layers
        for layer in range(self.num_layers - 1):
            out_act = []
            for node in self.mvtu_network[layer]:
                node_out = node.run(in_act)
                out_act.append(node_out)

            out_act = np.array(out_act)
            in_act = out_act.reshape(len(out_act), -1)

        return in_act

    def evaluate(self, test_data):
        # Form tuples of prediction and target class for all test samples
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        # Return number of predictions which match target class
        return sum(int(x == y) for (x, y) in test_results)

    def evaluate_one(self, test_data):
        (x, y) = test_data[0]
        pred = np.argmax(self.feedforward(x))
        print("Prediction: " + str(pred) + " Actual: " + str(y))
        return 0
