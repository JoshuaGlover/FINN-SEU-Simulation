from __future__ import division

import sys
import finn
import mnist_loader

NUM_ARGS = 4

# Check enough command line arguments are provided
if len(sys.argv) != NUM_ARGS:
    print("Usage: python weight_seu_test.py <hidden_layer_size> <num_seus> <num_tests>")
    sys.exit()

# Get command line args
hidden_layer_size = int(sys.argv[1])
num_seus          = int(sys.argv[2])
num_tests         = int(sys.argv[3])

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
