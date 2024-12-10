import torch
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def run_kmeans(n_samples, n_features, n_clusters, num_iterations):
    # 生成合成数据
    data, _ = make_blobs(n_samples=n_samples, 
                         centers=n_clusters,
                         cluster_std=0.60,
                         n_features=n_features,
                         random_state=0)
    tensor_data = torch.from_numpy(data).float()

    # 初始化质心
    centroids = tensor_data[torch.randperm(tensor_data.size(0))[:n_clusters]]

    for _ in range(num_iterations):
        # 计算距离
        distances = torch.cdist(tensor_data, centroids)

        # 分配标签
        _, labels = torch.min(distances, dim=1)

        # 更新质心
        for i in range(n_clusters):
            if torch.sum(labels == i) > 0:
                centroids[i] = torch.mean(tensor_data[labels == i], dim=0)

    # 可视化(仅前两维)
    if n_features >= 2:
        plt.scatter(data[:, 0], data[:, 1], c=labels.numpy(), cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='red')
        plt.title(f"K-Means Clustering (K={n_clusters})")
        plt.show()
    else:
        print("数据维度小于2维，无法进行二维可视化。")
