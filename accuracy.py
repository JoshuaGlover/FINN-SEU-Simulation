from __future__ import division

import sys
import finn
import mnist_loader

NUM_ARGS = 2

# Check enough command line arguments are provided
if len(sys.argv) != NUM_ARGS:
    print("Usage: python accuracy.py <hidden_layer_size>")
    sys.exit()

# Get hidden layer size
hidden_layer_size = int(sys.argv[1])

# Load MNIST dataset
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create network and calculate accuracy on test set
net = finn.FINN([784, hidden_layer_size, hidden_layer_size, 10])
accuracy = net.evaluate(test_data)/len(test_data)
print("Accuracy: " + str(accuracy*100) + "%")
