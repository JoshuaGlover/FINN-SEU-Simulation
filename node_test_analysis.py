from __future__ import division
import numpy as np

# Open log file
fp = open("results/inversion/stuck_low_test_64.txt", "r")

# Layer sizes
layer_sizes = [784, 64, 64, 10]
jumps_by_layer = [[0] * size for size in layer_sizes]
all_jumps = []

# Split log file by lines
log_lines = fp.read().splitlines()
for line in log_lines:
    jump, layer, node = line.split("\t")
    jump  = float(jump)
    layer = int(layer)
    node  = int(node)
    jumps_by_layer[layer][node] = jump
    all_jumps.append(jump)

for layer in range(len(layer_sizes)):
    mean = sum(jumps_by_layer[layer])/len(jumps_by_layer[layer])
    max  = np.max(jumps_by_layer[layer])
    print"Layer: {:} \t Mean: {:.2f}% \t Max: {:.2f}%".format(layer, mean, max)

mean_all = sum(all_jumps)/len(all_jumps)
print"Mean of All: {:2f}%".format(mean_all)
