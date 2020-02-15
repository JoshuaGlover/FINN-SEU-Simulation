"""
This script converts the trained weights and batch-norm parameters pickled by the BNN-PYNQ trainer.
Check out the project here: https://github.com/Xilinx/BNN-PYNQ/tree/master/bnn/src/training

Usage: python convert_params.py <hidden_layer_size> <parameter_file>
"""

import pickle
import sys
import argparse
import finnthesizer as fth
import numpy        as np


# Process command line arguments
parser = argparse.ArgumentParser(description="Converts params from BNN-PYNQ to NumPy arrays and pickles them")
parser.add_argument("hidden_layer_size", type=int, help="Size of hidden layer")
parser.add_argument("in_file", type=str, help="BNN-PYNQ file containing trained params")
args = parser.parse_args()

hidden_layer_size = args.hidden_layer_size
in_file = args.bnn_pynq_file

# Create finnthesizer reader
reader = fth.BNNWeightReader("params/" + in_file, False)

# Read weight and thresholds layer by layer
(w0, t0) = reader.readFCBNComplex(0, 0, 0, 1, 1, 1)
(w1, t1) = reader.readFCBNComplex(0, 0, 0, 1, 1, 1)
(w2, t2) = reader.readFCBNComplex(0, 0, 0, 1, 1, 1)

# Convert to numpy array
t0 = np.array(t0)
t1 = np.array(t1)
t2 = np.array(t2)

# Convert to 2D array
t0 = t0.reshape(hidden_layer_size, -1)
t1 = t1.reshape(hidden_layer_size, -1)
t2 = t2.reshape(10, -1)

# Store weights and thresholds in lists
weights    = [w0, w1, w2]
thresholds = [t0, t1, t2]

# Pickle weight and thresholds so they can be read by FINN BNN
out_file = "_" + str(hidden_layer_size) + "_" + str(hidden_layer_size)
with open("weights/weights" + out_file + ".txt", "wb") as fp:
    pickle.dump(weights, fp)
with open("thresholds/thresholds" + out_file + ".txt", "wb") as fp:
    pickle.dump(thresholds, fp)
