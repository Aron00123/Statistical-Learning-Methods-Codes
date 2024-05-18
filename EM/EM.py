import numpy as np
from scipy.stats import norm


def expectation(data, means, covariances, weights):
    num_components = len(means)
    num_points = len(data)
    responsibilities = np.zeros((num_points, num_components))

    for i in range(num_points):
        for j in range(num_components):
            responsibilities[i, j] = weights[j] * norm.pdf(data[i], loc=means[j], scale=np.sqrt(covariances[j]))

    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
    return responsibilities


def maximization(data, responsibilities):
    num_components = responsibilities.shape[1]
    num_points = len(data)

    total_weight = np.sum(responsibilities, axis=0)
    weights = total_weight / num_points
    means = np.sum(data[:, np.newaxis] * responsibilities, axis=0) / total_weight

    covariances = np.zeros(num_components)
    for j in range(num_components):
        diff = data - means[j]
        covariances[j] = np.sum(responsibilities[:, j] * diff ** 2) / total_weight[j]

    return means, covariances, weights


def EM(data, initial_means, initial_covariances, initial_weights, num_iterations):
    means = initial_means
    covariances = initial_covariances
    weights = initial_weights

    for i in range(num_iterations):
        responsibilities = expectation(data, means, covariances, weights)
        means, covariances, weights = maximization(data, responsibilities)

    return means, covariances, weights


# Generate synthetic data
np.random.seed(0)
data = np.concatenate([np.random.normal(-10, 1, 1000), np.random.normal(10, 1, 1000)])

# Initial guesses for parameters
initial_means = [-8, 8]
initial_covariances = [1, 1]
initial_weights = [0.4, 0.6]

# Run EM algorithm
estimated_means, estimated_covariances, estimated_weights = EM(data, initial_means, initial_covariances,
                                                               initial_weights, num_iterations=100)

print("Estimated means:", estimated_means)
print("Estimated covariances:", estimated_covariances)
print("Estimated weights:", estimated_weights)
