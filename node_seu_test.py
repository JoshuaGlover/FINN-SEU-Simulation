"""
This script performs a test in which, for each node, a new network is loaded and an upset is
simulated on a specific node. The resulting change in accuracy is recorded in a log file located in
the results/node_seu directory. This is then repeated for each node and every node in the network.

Inversion, stuck high or stuck low errors can be chosen as well as the size of the network based on
the command line arguments given.

Usage: python node_seu_test.py <hidden_layer_size> <-i or -l or -H>
"""

from __future__ import division
import sys
import finn
import argparse
import mnist_loader
import numpy as np

# Process command line arguments
parser = argparse.ArgumentParser(description="Systematic SEU test")
parser.add_argument("hidden_layer_size", type=int, help="Size of hidden layer")

group = parser.add_mutually_exclusive_group()
group.add_argument('-i', "--inversion", action='store_true', help="Use inversion for upsets")
group.add_argument('-l', "--stuck_low", action='store_true', help="Use stuck_low for upsets")
group.add_argument('-H', "--stuck_high", action='store_true', help="Use stuck_high for upsets")

args = parser.parse_args()

hidden_layer_size = args.hidden_layer_size
layer_sizes       = [784, hidden_layer_size, hidden_layer_size, 10]

# Setup log file
if args.inversion:
    mode = "inversion"
elif args.stuck_low:
    mode = "stuck_low"
elif args.stuck_high:
    mode = "stuck_high"
else:
    parser.error('No upset mode requested')

log_file_name = "results/node_seu/" + mode + "_test_" + str(hidden_layer_size) + ".txt"
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
        # Create new network
        net = finn.FINN(layer_sizes)

        # Perform correct SEU
        if args.inversion:
            net.node_inversion_at(layer, node)
            accuracy = net.evaluate(test_data, inversion=True)/len(test_data)
        elif args.stuck_low:
            net.node_stuck_high_at(layer, node)
            accuracy = net.evaluate(test_data, stuck_high=True)/len(test_data)
        elif args.stuck_high:
            net.node_stuck_low_at(layer, node)
            accuracy = net.evaluate(test_data, stuck_low=True)/len(test_data)

        print("Accuracy: " + str(accuracy * 100) + "%")
        jump = "{:.2f}".format(np.abs(start_acc - accuracy)*100)
        log_line = "{} \t {} \t {}".format(jump, str(layer), str(node))
        fp.write(log_line + "\n")
        # print"Jump: {}%".format(str(np.abs(start_acc - accuracy)*100))
