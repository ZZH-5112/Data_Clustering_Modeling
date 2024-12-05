import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Generate synthetic data
X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)
X = torch.tensor(X, dtype=torch.float)

def euclidean_distance(x1, x2):
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))


def dbscan(X, eps, min_samples):
  
    n_samples = X.shape[0]
    labels = torch.zeros(n_samples, dtype=torch.int)

    # Initialize cluster label and visited flags
    cluster_label = 0
    visited = torch.zeros(n_samples, dtype=torch.bool)

    # Iterate over each point
    for i in range(n_samples):
        if visited[i]:
            continue
        visited[i] = True

        # Find neighbors
        neighbors = torch.nonzero(euclidean_distance(X[i], X) < eps).squeeze()
        
        if neighbors.shape[0] < min_samples:
            # Label as noise
            labels[i] = 0
        else:
            # Expand cluster
            cluster_label += 1
            labels[i] = cluster_label
            expand_cluster(X, labels, visited, neighbors, cluster_label, eps, min_samples)

    return labels


def expand_cluster(X, labels, visited, neighbors, cluster_label, eps, min_samples):
    i = 0
    while i < neighbors.shape[0]:
        neighbor_index = neighbors[i].item()
        if not visited[neighbor_index]:
            visited[neighbor_index] = True
            neighbor_neighbors = torch.nonzero(euclidean_distance(X[neighbor_index], X) < eps).squeeze()
            if neighbor_neighbors.shape[0] >= min_samples:
                neighbors = torch.cat((neighbors, neighbor_neighbors))
        if labels[neighbor_index] == 0:
            labels[neighbor_index] = cluster_label
        i += 1

# DBSCAN parameters
eps = 0.3
min_samples = 5

# Perform clustering
labels = dbscan(X, eps, min_samples)

# Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()