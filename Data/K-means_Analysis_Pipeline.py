import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm

def kmeans_analysis_pipeline(csv_file, k_range, iteration_range):
    # 读取和预处理数据
    data = pd.read_csv(csv_file)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 参数搜索
    results = []
    for k in tqdm(k_range, desc="K值"):
        for iterations in iteration_range:
            kmeans = KMeans(n_clusters=k, max_iter=iterations, n_init=10, random_state=42)
            labels = kmeans.fit_predict(scaled_data)
            
            silhouette_avg = silhouette_score(scaled_data, labels)
            inertia = kmeans.inertia_
            
            results.append({
                'k': k,
                'iterations': iterations,
                'silhouette_score': silhouette_avg,
                'inertia': inertia
            })
    
    # 找出最佳参数
    best_result = max(results, key=lambda x: x['silhouette_score'])
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # Elbow Method
    plt.subplot(121)
    for iter in iteration_range:
        inertias = [r['inertia'] for r in results if r['iterations'] == iter]
        plt.plot(k_range, inertias, label=f'Iterations={iter}')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.legend()
    
    # Silhouette Analysis
    plt.subplot(122)
    for iter in iteration_range:
        silhouette_scores = [r['silhouette_score'] for r in results if r['iterations'] == iter]
        plt.plot(k_range, silhouette_scores, label=f'Iterations={iter}')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 使用最佳参数进行聚类
    best_kmeans = KMeans(n_clusters=best_result['k'], max_iter=best_result['iterations'], n_init=10, random_state=42)
    best_labels = best_kmeans.fit_predict(scaled_data)
    
    # 可视化最佳聚类结果
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=best_labels, cmap='viridis')
    plt.scatter(best_kmeans.cluster_centers_[:, 0], best_kmeans.cluster_centers_[:, 1], 
                marker='x', s=200, linewidths=3, color='r', zorder=10)
    plt.title(f'K-Means Clustering (K={best_result["k"]})')
    plt.colorbar(scatter)
    plt.show()
    
    print(f"最佳参数: K={best_result['k']}, Iterations={best_result['iterations']}")
    print(f"最佳轮廓系数: {best_result['silhouette_score']:.4f}")
    
    return best_result, results

# 使用示例
# csv_file = 'your_data.csv'
# k_range = range(5, 21)
# iteration_range = [100, 200, 300]
# best_result, all_results = kmeans_analysis_pipeline(csv_file, k_range, iteration_range)