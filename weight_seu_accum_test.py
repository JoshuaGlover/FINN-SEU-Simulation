"""
This script performs weight SEU accumulation tests. Each test consists of performing a random
weight flip and recording the resulting accuracy to a log file in results/weight_seu. This is repeated
a set number of times per test and multiple tests can be performed.

Usage: python weight_seu_accum_test.py <hidden_layer_size> <num_seus> <num_tests>
"""

from __future__ import division
import sys
import finn
import argparse
import mnist_loader

# Process command line arguments
parser = argparse.ArgumentParser(description="Reports change in accuracy after multiple SEUs")
parser.add_argument("hidden_layer_size", type=int, help="Size of hidden layer")
parser.add_argument("num_seus", type=int, help="Number of SEUs per test")
parser.add_argument("num_tests", type=int, help="Number of tests")
args = parser.parse_args()

hidden_layer_size = args.hidden_layer_size
num_seus  = args.num_seus
num_tests = args.num_tests

# Load MNIST dataset
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

for num_test in range(num_tests):
    # Setup log file
    log_file_name = "results/weight_seu/seu_test_" + str(hidden_layer_size) + "-" + str(num_test) + ".txt"
    fp = open(log_file_name, "w")

    # Create network and calculate accuracy on test set
    net = finn.FINN([784, hidden_layer_size, hidden_layer_size, 10])

    # Print starting accuracy
    accuracy = net.evaluate(test_data)/len(test_data)
    print("Starting Accuracy: " + str(accuracy*100) + "%")

    for seu in range(num_seus):
        log_info = net.weight_seu()
        accuracy = net.evaluate(test_data)/len(test_data) * 100
        fp.write(str(accuracy) + log_info + "\n")
        print("Accuracy: " + str(accuracy) + "%")

    # Print finishing accuracy
    print("Finishing Accuracy: " + str(accuracy) + "%")
