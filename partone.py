import tensorflow as tf
import numpy as np

inputs = [6, 7, 3, 2, 4]
weights = [0.3, 0.1, 0.5, 0.6, 0.1]
bias = -1

weighted_inputs = [inputs[i] * weights[i] for i in range(len(inputs))]
total_input = sum(weighted_inputs) + bias


def tanh(x):
    return np.tanh(x)


activated_output = tanh(total_input)
print("Explorer mood level:", activated_output)