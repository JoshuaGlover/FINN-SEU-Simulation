from __future__ import division

import finn
import finn_mvtu
import mvtu
import sys
import mnist_loader
import numpy as np

# a = np.array([[1,0,1],[1,1,0]])
# b = np.array([[1],[0],[1]])
#
# print(a)
# print(b)
#
# out = finn.dot_xnor(a, b)
#
# print(out)
#
# mvtu = mvtu.MVTU(a[0,:], 2)
# activation = mvtu.run(b)
#
# print(activation)

NUM_ARGS = 2

# Check enough command line arguments are provided
if len(sys.argv) != NUM_ARGS:
    print("Usage: python test.py <hidden_layer_size>")
    sys.exit()

# Get hidden layer size
hidden_layer_size = int(sys.argv[1])

# Load MNIST dataset
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create network and calculate accuracy on test set
net = finn_mvtu.FINN([784, hidden_layer_size, hidden_layer_size, 10])
accuracy = net.evaluate(test_data)/len(test_data)
print("Accuracy: " + str(accuracy*100) + "%")
