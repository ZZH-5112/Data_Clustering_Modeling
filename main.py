import argparse
from K_means_Clustering import run_kmeans


def main():
    parser = argparse.ArgumentParser(description="Run various clustering algorithms with given parameters.")

    parser.add_argument("--algorithm", type=str, default="kmeans", 
                        help="Name of the clustering algorithm to run (kmeans/dbscan/agglomerative)")

    # 通用参数
    parser.add_argument("--n_samples", type=int, default=3000, help="Number of samples")
    parser.add_argument("--n_features", type=int, default=2, help="Number of features for each sample")

    # 针对K-Means的参数
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters (K for K-Means and Agglomerative)")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations for K-Means")


    args = parser.parse_args()

    if args.algorithm.lower() == "kmeans":
        run_kmeans(n_samples=args.n_samples,
                   n_features=args.n_features,
                   n_clusters=args.n_clusters,
                   num_iterations=args.num_iterations)

    else:
        print(f"Unknown algorithm: {args.algorithm}. ")


if __name__ == "__main__":
    main()
