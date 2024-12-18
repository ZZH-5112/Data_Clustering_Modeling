# Data Clustering Modeling

## Overview

This project provides implementations of three commonly used unsupervised clustering algorithms: **K-Means**, **Hierarchical Clustering**, and **DBSCAN**. These algorithms are implemented in Python using the PyTorch framework and are applied to synthetic or benchmark datasets to demonstrate their effectiveness.

### What is Unsupervised Clustering?
Unsupervised clustering is a fundamental machine learning technique that groups data based on similarity or underlying structure without requiring labeled data. Popular clustering methods include:

1. **K-Means Clustering**: A partitioning algorithm that divides data points into K clusters, where each cluster is represented by the mean of its data points.
2. **Hierarchical Clustering**: Constructs a hierarchy of clusters using either a bottom-up (agglomerative) or top-down (divisive) approach.
3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: A density-based clustering method capable of discovering clusters of arbitrary shapes and effectively handling noise.

## Features
- **K-Means Clustering**: Implementation using PyTorch tensors and visualization.
- **Hierarchical Clustering**: Implementation using PyTorch and SciPy with dendrogram visualization.
- **DBSCAN Clustering**: Implementation using PyTorch for density-based clustering, supporting flexibility and noise resistance.

## Dependencies
Before running the project, ensure the following Python libraries are installed:

- `torch`: For tensor operations and computations.
- `scikit-learn`: For dataset generation and clustering evaluation metrics.
- `matplotlib`: For data visualization.
- `scipy`: For hierarchical clustering.

## Installation Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ZZH-5112/Data_Clustering_Modeling.git
   cd Data_Clustering_Modeling
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Each clustering algorithm is implemented in a separate Python script:

- `K-means_Clustering.py`: Implements the K-Means clustering algorithm.
- `Agglomerative_Clustering.py`: Implements hierarchical agglomerative clustering.
- `DBSCAN_Clustering.py`: Implements the DBSCAN clustering algorithm.

This project provides a unified entry point through `main.py`, allowing you to specify the clustering algorithm and its parameters via command-line arguments. Currently, the supported algorithms are **K-Means**, **Hierarchical/Agglomerative Clustering**, and **DBSCAN**.

### Example Usage

- **Running K-Means**:
  ```bash
  python main.py --file_path Data/CSV_Data/Data_samples300_clusters5.csv \
                 --algorithm kmeans \
                 --n_clusters 7 \
                 --num_iterations 300
              
  ```
  Parameter description:
  - `--file_path`: the root of data.
  - `--algorithm kmeans`: Specifies the algorithm to run as K-Means.
  - `--n_clusters 7`: Number of clusters to form.
  - `--num_iterations 300`: Number of iterations for centroid updates.
  - `--selected_features 1 2 4`: Select the feature columns to use (in this example, columns 1, 2, and 4 are selected). If not specified, all features are used by default.
  - `--dimensionality_reduction`: Enable dimensionality reduction processing.

If you wish to extend the project with additional algorithms or parameters, you only need to implement the logic in the respective algorithm script and add the corresponding argument parsing and function call in `main.py`.

### Running Scripts Directly (Non-Parameterized)
If you prefer to run individual clustering scripts directly (e.g., `K_means_Clustering.py`), you can do so by executing:

```bash
python K_means_Clustering.py
```
The same applies to other scripts like `Agglomerative_Clustering.py` and `DBSCAN_Clustering.py`. However, it is recommended to use `main.py` for a more flexible and parameterized approach.

## Algorithms Overview

### K-Means Clustering
- **Concept**: Partitions data into K clusters by minimizing the sum of squared distances within clusters.
- **Implementation Highlights**:
  - Randomly initializes centroids.
  - Iteratively assigns data points to the nearest centroid and updates centroids.
  - Visualizes clusters and centroids.

### Hierarchical Clustering
- **Concept**: Builds a hierarchy of clusters through repeated merging or splitting.
- **Implementation Highlights**:
  - Uses PyTorch for pairwise distance computation.
  - Converts the distance matrix to NumPy format for SciPy's linkage function.
  - Visualizes the resulting dendrogram.

### DBSCAN Clustering
- **Concept**: Groups densely packed points and labels low-density points as noise.
- **Implementation Highlights**:
  - Implements the core DBSCAN logic using PyTorch tensors.
  - Assigns clusters based on density and proximity.
  - Visualizes clustering results with matplotlib.

## Evaluating Clustering Performance
To assess the quality of clustering results, the following metrics can be used:

- **Silhouette Score**: Measures the compactness and separation of clusters.
- **Davies-Bouldin Index**: Evaluates cluster similarity and cluster spread.
- **Calinski-Harabasz Index**: Ratio of within-cluster dispersion to between-cluster dispersion.

These metrics can be integrated into the scripts for automated evaluation.

## Results
Each script performs the following steps:
1. Loads or generates a dataset.
2. Applies the respective clustering algorithm.
3. Visualizes the results.

## References
- [GeeksforGeeks: PyTorch for Unsupervised Clustering](https://www.geeksforgeeks.org/pytorch-for-unsupervised-clustering/)
- Official PyTorch Documentation

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.