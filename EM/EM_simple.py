import numpy as np


def expectation(data, bias):
    heads_prob = bias
    tails_prob = 1 - bias
    return data * heads_prob / (data * heads_prob + (1 - data) * tails_prob)


def maximization(data, responsibilities):
    return np.sum(responsibilities) / len(data)


def EM(data, initial_bias, num_iterations):
    bias = initial_bias

    for i in range(num_iterations):
        responsibilities = expectation(data, bias)
        bias = maximization(data, responsibilities)

    return bias


# Generate synthetic data
np.random.seed(0)
data = np.random.randint(0, 2, size=1000000)

# Initial guess for bias
initial_bias = 0.5

# Run EM algorithm
estimated_bias = EM(data, initial_bias, num_iterations=1)

print("Estimated bias:", estimated_bias)
