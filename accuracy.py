"""
This script measures and reports the accuracy of the network matching the hidden layer size given.

Usage: python accuracy.py <hidden_layer_size>

Weights and thresholds for a network with hidden_layer_size x must be stored in the weights and
thresholds directory as weights_x_x.txt and thresholds_x_x.txt respectively. This project supports
networks with two equally sized hidden layers only.
"""

from __future__ import division
import sys
import finn
import argparse
import mnist_loader

# Process command line arguments
parser = argparse.ArgumentParser(description="Measures Accuracy of Network")
parser.add_argument("hidden_layer_size", type=int, help="Hidden Layer Size of Network")
args = parser.parse_args()

# Load MNIST dataset
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create network and calculate accuracy on test set
net = finn.FINN([784, args.hidden_layer_size, args.hidden_layer_size, 10])
accuracy = net.evaluate(test_data)/len(test_data)

print("Accuracy: " + str(accuracy*100) + "%")
