import torch
import pandas as pd
import numpy as np
import os
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

'''
行：样本数
列：多参数（目前每个参数一维）+id列
生成一个csv文件
样本数300、500、1000

进行测试，使用20维
根据数据的维度对于n_clusters和num_iterations

生成3个csv文件
读取csv文件的时候，可以指定维度（选定）

对5个特征进行加权求和取平均得到一个新的特征值（特征降维）

'''

def load_and_preprocess_data(file_path, selected_features=None, perform_dimensionality_reduction=False):
    """从CSV读取数据，选择特征，可能进行特征降维"""
    df = pd.read_csv(file_path)
   
    if selected_features is None:
        selected_features = list(range(1, len(df.columns)))
    
    data = df.iloc[:, selected_features].values
   
    if perform_dimensionality_reduction:
        # 示例：对每5个特征进行加权求和取平均
        n_features = data.shape[1]
        new_data = []
        for i in range(0, n_features, 5):
            new_feature = np.mean(data[:, i:i+5], axis=1)
            new_data.append(new_feature)
        data = np.array(new_data).T
   
    # # 标准化数据
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    
    return data

def perform_kmeans_clustering(data, n_clusters, num_iterations):
    """执行K-means聚类算法"""
    tensor_data = torch.from_numpy(data).float()
    centroids = tensor_data[torch.randperm(tensor_data.size(0))[:n_clusters]]

    for _ in range(num_iterations):
        distances = torch.cdist(tensor_data, centroids)
        _, labels = torch.min(distances, dim=1)
        for i in range(n_clusters):
            if torch.sum(labels == i) > 0:
                centroids[i] = torch.mean(tensor_data[labels == i], dim=0)

    return labels.numpy(), centroids.numpy()

def visualize_clustering(data, labels, centroids):
    """绘制聚类结果的散点图"""
    if data.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='red')
        plt.title(f"K-Means Clustering (K={len(centroids)})")
        plt.show()
    else:
        print("数据维度小于2维，无法进行二维可视化。")

def run_kmeans(file_path, n_clusters, num_iterations, selected_features=None, perform_dimensionality_reduction=False):
    """完整的K-means聚类分析流程"""
    # 加载和预处理数据
    data = load_and_preprocess_data(file_path, selected_features, perform_dimensionality_reduction)

    print(data.shape)
    
    # 执行K-means聚类
    labels, centroids = perform_kmeans_clustering(data, n_clusters, num_iterations)
    
    # 可视化结果
    visualize_clustering(data, labels, centroids)
    
    return labels, centroids

# 使用示例
if __name__ == "__main__":
    file_path = "Data/CSV_Data/Data_samples300_clusters5.csv"
    n_clusters = 7
    num_iterations = 100
    
    labels, centroids = run_kmeans(file_path, n_clusters, num_iterations)
    print(f"聚类完成。共{n_clusters}个簇。")
