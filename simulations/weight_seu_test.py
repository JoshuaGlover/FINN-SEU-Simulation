from __future__ import division

import sys
import finn
import mnist_loader
import numpy as np

NUM_ARGS = 2

# Check enough command line arguments are provided
if len(sys.argv) != NUM_ARGS:
    print("Usage: python weight_seu_test.py <hidden_layer_size>")
    sys.exit()

# Get command line args
hidden_layer_size = int(sys.argv[1])
layer_sizes       = [784, hidden_layer_size, hidden_layer_size, 10]

# Setup log file
log_file_name = "results/last_layer/weight_seu_test_" + str(hidden_layer_size) + ".txt"
fp = open(log_file_name, "w")

# Load MNIST dataset
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create network and calculate accuracy on test set
net = finn.FINN(layer_sizes)

# Print starting accuracy
start_acc = net.evaluate(test_data)/len(test_data)
print("No Error Accuracy: " + str(start_acc * 100) + "%")

# for layer in range(3):
layer = 2
for node_to in range(layer_sizes[layer + 1]):
    for node_from in range(layer_sizes[layer]):
        # Create fresh network
        net = finn.FINN(layer_sizes)
        net.weight_seu_at(layer, node_to, node_from)
        accuracy = net.evaluate(test_data)/len(test_data)
        log_line = "{} \t {} \t {}".format(str(np.abs(start_acc - accuracy)), str(node_to), str(node_from))
        fp.write(log_line + "\n")
        print"Weight Flip at [{}][{}][{}] \t Accuracy: {}%".format(layer, node_to, node_from, str(accuracy * 100))
