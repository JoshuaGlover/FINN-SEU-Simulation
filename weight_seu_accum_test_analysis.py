from __future__ import division

import pickle
import numpy as np

hidden_layer_size = 64
num_tests         = 9
num_seus          = 1000

all_logs = [[] * num_tests for _ in range(num_tests)]

for num_test in range(num_tests):
    # Open log file
    log_file_name = "results/weight_seu/seu_test_" + str(hidden_layer_size) + "-" + str(num_test) + ".txt"
    fp = open(log_file_name, "r")

    # Split log file by lines
    lines = fp.read().splitlines()
    for line in lines:
        line_split = line.split("\t")
        acc = float(line_split[0])
        all_logs[num_test].append(acc)

mean_of_logs = []

for num_seu in range(num_seus):
    mean = 0
    for num_test in range(num_tests):
        mean += all_logs[num_test][num_seu]

    mean = mean/num_tests
    mean_of_logs.append(mean)

fp = open("mean_log.txt", "w")
for num_seu in range(num_seus):
    fp.write(str(mean_of_logs[num_seu]) + "\n")
