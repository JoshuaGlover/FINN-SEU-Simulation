from __future__ import division

import sys
import finn
import mnist_loader
import numpy as np

NUM_ARGS = 3

# Check enough command line arguments are provided
if len(sys.argv) != NUM_ARGS:
    print("Usage: python node_inversion_test.py <hidden_layer_size> <mode>")
    sys.exit()

# Get command line args
hidden_layer_size = int(sys.argv[1])
layer_sizes       = [784, hidden_layer_size, hidden_layer_size, 10]
mode              = sys.argv[2]

# Setup log file
log_file_name = "results/inversion/" + mode + "_test_" + str(hidden_layer_size) + ".txt"
fp = open(log_file_name, "w")

# Load MNIST dataset
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create network and calculate accuracy on test set
net = finn.FINN(layer_sizes)

# Print starting accuracy
start_acc = net.evaluate(test_data)/len(test_data)
print("No Error Accuracy: " + str(start_acc * 100) + "%")

for layer in range(len(layer_sizes)):
    for node in range(layer_sizes[layer]):
        # Create fresh network
        net = finn.FINN(layer_sizes)

        # Perform correct SEU
        if mode == "inversion":
            net.node_inversion_at(layer, node)
            accuracy = net.evaluate(test_data, inversion=True)/len(test_data)
        elif mode == "stuck_high":
            net.node_stuck_high_at(layer, node)
            accuracy = net.evaluate(test_data, stuck_high=True)/len(test_data)
        elif mode == "stuck_low":
            net.node_stuck_low_at(layer, node)
            accuracy = net.evaluate(test_data, stuck_low=True)/len(test_data)

        print("Accuracy: " + str(accuracy * 100) + "%")
        jump = "{:.2f}".format(np.abs(start_acc - accuracy)*100)
        log_line = "{} \t {} \t {}".format(jump, str(layer), str(node))
        fp.write(log_line + "\n")
        # print"Jump: {}%".format(str(np.abs(start_acc - accuracy)*100))
