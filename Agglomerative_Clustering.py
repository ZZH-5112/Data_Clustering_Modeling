import torch
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Sample data
X = torch.tensor([[1, 2], [1, 4], [1, 0],
                 [4, 2], [4, 4], [4, 0]])

# Standardize data (ensure floating point output)
X_std = (X.float() - X.float().mean(dim=0)) / X.float().std(dim=0)

# Calculate pairwise Euclidean distances using PyTorch
distances = torch.cdist(X_std, X_std, p=2)  # p=2 for Euclidean distance

# Convert distances to numpy array for SciPy usage
distances = distances.numpy()

# Perform hierarchical clustering using SciPy
Z = linkage(distances, 'single')

# Plot dendrogram using matplotlib
plt.figure(figsize=(10, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
dendrogram(Z)
plt.show()