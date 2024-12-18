import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import os

def generate_and_save_data(n_samples, n_features, n_clusters, output_dir, filename):
    """生成合成数据并保存为CSV"""
    data, _ = make_blobs(n_samples=n_samples, centers=n_clusters,
                         cluster_std=0.60, n_features=n_features, random_state=0)
   
    df = pd.DataFrame(data, columns=[f'feature_{i+1}' for i in range(n_features)])
    df.insert(0, 'id', range(1, len(df) + 1))
   
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)
   
    print(f"数据已生成并保存到: {file_path}")
    return file_path

# 使用示例
if __name__ == "__main__":
    n_samples = 1000
    n_features = 20
    n_clusters = 12
    output_dir = "Data/CSV_Data"
    filename = "Data_samples1000_clusters12.csv"
    
    generate_and_save_data(n_samples, n_features, n_clusters, output_dir, filename)
