from __future__ import division
import numpy as np

# Open log file
fp = open("results/last_layer/weight_seu_test_64.txt", "r")

# List of jumps by digits
jumps_by_dig = [[] * 10 for _ in range(10)]

# Split log file by lines
lines = fp.read().splitlines()
for line in lines:
    # Split line by tabs
    line_split = line.split("\t")
    jump  = float(line_split[0]) * 100
    digit = int(line_split[1])
    jumps_by_dig[digit].append(jump)

# Calculate mean jump for each digit
for (digit, digit_jumps) in enumerate(jumps_by_dig):
    print"Digit: {:}, Mean Jump: {:.2f}, Max Jump: {:.2f}%".format(digit, \
        sum(digit_jumps)/len(digit_jumps), max(digit_jumps))
