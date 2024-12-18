import argparse
from K_means_Clustering import run_kmeans


def main():
    parser = argparse.ArgumentParser(description="Run various clustering algorithms with given parameters.")

    parser.add_argument("--algorithm", type=str, default="kmeans", 
                        help="Name of the clustering algorithm to run (kmeans/dbscan/agglomerative)")

    # 通用参数
    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--selected_features", type=int, nargs='*', help="List of feature indices to use (e.g., 1 2 4)")
    parser.add_argument("--dimensionality_reduction", action="store_true", help="Perform dimensionality reduction")
    

    # 针对K-Means的参数
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters (K for K-Means and Agglomerative)")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations for K-Means")


    args = parser.parse_args()

    selected_features = args.selected_features if args.selected_features else None


    if args.algorithm.lower() == "kmeans":
        run_kmeans(file_path=args.file_path,
                   n_clusters=args.n_clusters,
                   num_iterations=args.num_iterations,
                   selected_features=selected_features,
                   perform_dimensionality_reduction=args.dimensionality_reduction
                   )

    else:
        print(f"Unknown algorithm: {args.algorithm}. ")


if __name__ == "__main__":
    main()
