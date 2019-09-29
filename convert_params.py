import pickle
import sys
import finnthesizer as fth
import numpy        as np

NUM_ARGS = 3

# Check enough command line arguments are provided
if len(sys.argv) != NUM_ARGS:
    print("Usage: python convert_params.py <hidden_layer_size> <param file suffix>")
    sys.exit()

# Get file ending for target parameter file
hidden_layer_size = int(sys.argv[1])
in_file           = sys.argv[2]

# Create finnthesizer reader
reader = fth.BNNWeightReader("params/mnist_1w_1a_" + in_file + ".npz", False)

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
out_file = "_" + str(hidden_layer_size) + "_" + in_file
with open("weights/weights" + out_file + ".txt", "wb") as fp:
    pickle.dump(weights, fp)
with open("thresholds/thresholds" + out_file + ".txt", "wb") as fp:
    pickle.dump(thresholds, fp)
