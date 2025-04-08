import subprocess
import sys
import pandas as pd
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import defaultdict


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


class KdTree:
    def __init__(self, dim, points):
        self._dim = dim
        self._points = points.copy()  # Creating a copy to avoid modifying the original
        random.shuffle(self._points)  # Shuffling the dataset to ensure balanced splits
        self._root = self._make_kd_tree()

    def _make_kd_tree(self):
        if not self._points:
            return None
        return self._make_kd_tree_rec(self._points, 0)

    def _make_kd_tree_rec(self, points, depth=0):
        if not points:
            return None
        
        axis = depth % self._dim
        points.sort(key=lambda x: x[axis])
        
        median_idx = len(points) // 2
        
        return [
            self._make_kd_tree_rec(points[:median_idx], depth + 1),
            self._make_kd_tree_rec(points[median_idx + 1:], depth + 1),
            points[median_idx]
        ]

    def print_kd_tree(self, node=None, depth=0, max_depth=5):
        """
        Print a visual representation of the KD-tree structure.
        Limited to a specified max_depth to avoid overwhelming output.
        """
        if node is None:
            node = self._root
        
        if node is None:
            return
            
        # Indentation for tree depth visualization
        indent = "  " * depth
        
        # Only print up to max_depth levels
        if depth <= max_depth:
            axis = depth % self._dim
            axis_name = ["l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice"][axis]
            print(f"{indent}Depth {depth} (split on {axis_name}): {node[2]}")
            
            if depth == max_depth:
                print(f"{indent}  ... (tree continues)")
            else:
                self.print_kd_tree(node[0], depth + 1, max_depth)
                self.print_kd_tree(node[1], depth + 1, max_depth)
        
    def tree_stats(self):
        """Calculate and return statistics about the KD-tree"""
        stats = {
            'total_nodes': 0,
            'max_depth': 0,
            'leaf_nodes': 0,
            'internal_nodes': 0,
            'depth_counts': {}
        }
        
        self._tree_stats_rec(self._root, 0, stats)
        
        # Calculate average depth
        total_depth = sum(depth * count for depth, count in stats['depth_counts'].items())
        stats['avg_depth'] = total_depth / stats['total_nodes'] if stats['total_nodes'] > 0 else 0
        
        return stats
        
    def _tree_stats_rec(self, node, depth, stats):
        if node is None:
            return
            
        stats['total_nodes'] += 1
        stats['max_depth'] = max(stats['max_depth'], depth)
        
        # Count nodes at each depth
        if depth not in stats['depth_counts']:
            stats['depth_counts'][depth] = 0
        stats['depth_counts'][depth] += 1
        
        # Check if this is a leaf node
        if node[0] is None and node[1] is None:
            stats['leaf_nodes'] += 1
        else:
            stats['internal_nodes'] += 1
            
        # Recurse on children
        self._tree_stats_rec(node[0], depth + 1, stats)
        self._tree_stats_rec(node[1], depth + 1, stats)

    def get_points_within_bounds(self, bounds):
        if not self._root:
            return []
        return self._get_points_within_bounds_rec(self._root, bounds)

    def _get_points_within_bounds_rec(self, kd_node, bounds, depth=0):
        if kd_node is None:
            return []
        
        results = []
        axis = depth % self._dim
        
        # Check if the current point is within bounds
        if kd_node[2] is not None and self._is_point_within_bounds(kd_node[2], bounds):
            results.append(kd_node[2])
        
        # Check if the splitting plane intersects the query box
        # If so, we need to check both left and right subtrees
        if kd_node[0] is not None and (kd_node[2] is None or bounds[axis][0] <= kd_node[2][axis]):
            results.extend(self._get_points_within_bounds_rec(kd_node[0], bounds, depth + 1))
        
        if kd_node[1] is not None and (kd_node[2] is None or bounds[axis][1] >= kd_node[2][axis]):
            results.extend(self._get_points_within_bounds_rec(kd_node[1], bounds, depth + 1))
        
        return results

    # Brute-force search for comparison
    def brute_force_search(self, bounds):
        return [point for point in self._points if self._is_point_within_bounds(point, bounds)]

    def _is_point_within_bounds(self, point, bounds):
        return all(bounds[i][0] <= point[i] <= bounds[i][1] for i in range(self._dim))


class OneDimensionalIndices:
    """
    Creates one-dimensional indices for each dimension using hash-based approach
    for faster range queries on individual dimensions.
    """
    def __init__(self, dim, points):
        self._dim = dim
        self._points = points
        self._indices = []
        
        # Create indices for each dimension
        for d in range(dim):
            index = defaultdict(list)
            for point in points:
                # Group points by their value in this dimension
                # Using rounding for floating point keys to create buckets
                # This is important for dimensions like price and quantity
                if isinstance(point[d], float):
                    # Adjust precision based on the dimension range
                    bucket_key = round(point[d], 1)  # Adjust precision as needed
                else:
                    bucket_key = point[d]
                index[bucket_key].append(point)
            self._indices.append(index)
    
    def get_points_within_bounds(self, bounds):
        """
        Performs a range query using the one-dimensional indices.
        Strategy: Use the most selective dimension to filter first, then check other constraints.
        """
        # Compute the selectivity of each dimension
        selectivity = []
        for d in range(self._dim):
            dimension_range = bounds[d][1] - bounds[d][0]
            # Calculate approximate selectivity (lower is more selective)
            if isinstance(bounds[d][0], float):
                # Get distinct values within range (using buckets)
                distinct_vals = sum(1 for k in self._indices[d].keys() 
                                  if bounds[d][0] <= k <= bounds[d][1])
                total_distinct = len(self._indices[d])
                selectivity.append(distinct_vals / max(1, total_distinct))
            else:
                # For integer dimensions, use the range directly
                distinct_vals = len(range(int(bounds[d][0]), int(bounds[d][1]) + 1))
                total_distinct = len(set(p[d] for p in self._points))
                selectivity.append(distinct_vals / max(1, total_distinct))
        
        # Choose the most selective dimension (lowest selectivity)
        primary_dim = selectivity.index(min(selectivity))
        
        # Get candidates from the most selective dimension
        candidates = []
        for key, points in self._indices[primary_dim].items():
            if bounds[primary_dim][0] <= key <= bounds[primary_dim][1]:
                candidates.extend(points)
        
        # Filter candidates against all other dimensions
        results = []
        for point in candidates:
            if all(bounds[d][0] <= point[d] <= bounds[d][1] for d in range(self._dim)):
                results.append(point)
        
        return results


def compare_search_methods(dataset, bounds, verbose=True):
    """
    Compare the performance of KD-Tree, One-Dimensional Indices, and Brute Force search.
    
    Parameters:
    - dataset: List of data points
    - bounds: List of (min, max) bounds for each dimension
    - verbose: Whether to print detailed results
    
    Returns:
    - Dictionary with timing results and match counts
    """
    dimensions = len(bounds)
    
    # Create a copy of the dataset for each method to ensure fair comparison
    dataset_copy1 = dataset.copy()
    dataset_copy2 = dataset.copy()
    dataset_copy3 = dataset.copy()
    
    # Setup KD-Tree
    print("Building KD-Tree...")
    start = time.time()
    kd_tree = KdTree(dimensions, dataset_copy1)
    kd_build_time = time.time() - start
    
    # Setup One-Dimensional Indices
    print("Building One-Dimensional Indices...")
    start = time.time()
    one_d_indices = OneDimensionalIndices(dimensions, dataset_copy2)
    one_d_build_time = time.time() - start
    
    # Define a brute force search function (no build time)
    def brute_force(points, query_bounds):
        return [p for p in points if all(query_bounds[i][0] <= p[i] <= query_bounds[i][1] 
                                        for i in range(len(query_bounds)))]
    
    # Execute KD-Tree search
    print("Executing KD-Tree search...")
    start = time.time()
    kd_results = kd_tree.get_points_within_bounds(bounds)
    kd_query_time = time.time() - start
    
    # Execute One-Dimensional Indices search
    print("Executing One-Dimensional Indices search...")
    start = time.time()
    one_d_results = one_d_indices.get_points_within_bounds(bounds)
    one_d_query_time = time.time() - start
    
    # Execute Brute Force search
    print("Executing Brute Force search...")
    start = time.time()
    brute_force_results = brute_force(dataset_copy3, bounds)
    brute_force_time = time.time() - start
    
    # Verify results
    kd_set = set(map(tuple, kd_results))
    one_d_set = set(map(tuple, one_d_results))
    bf_set = set(map(tuple, brute_force_results))
    
    results_match = (kd_set == bf_set == one_d_set)
    
    if verbose:
        print("\nSearch Results:")
        print(f"KD-Tree found {len(kd_results)} records in {kd_query_time:.4f}s (build: {kd_build_time:.4f}s)")
        print(f"One-D Index found {len(one_d_results)} records in {one_d_query_time:.4f}s (build: {one_d_build_time:.4f}s)")
        print(f"Brute Force found {len(brute_force_results)} records in {brute_force_time:.4f}s")
        
        if results_match:
            print("All methods returned identical results!")
        else:
            print("WARNING: Results differ between methods!")
            print(f"KD-Tree unique results: {len(kd_set)}")
            print(f"One-D Index unique results: {len(one_d_set)}")
            print(f"Brute Force unique results: {len(bf_set)}")
            print(f"Missing from KD-Tree vs Brute Force: {len(bf_set - kd_set)}")
            print(f"Missing from One-D Index vs Brute Force: {len(bf_set - one_d_set)}")
    
    return {
        'kd_build_time': kd_build_time,
        'kd_query_time': kd_query_time,
        'one_d_build_time': one_d_build_time,
        'one_d_query_time': one_d_query_time,
        'brute_force_time': brute_force_time,
        'kd_matches': len(kd_results),
        'one_d_matches': len(one_d_results),
        'brute_force_matches': len(brute_force_results),
        'results_match': results_match
    }


def visualize_results(query_results):
    """
    Create visualizations comparing performance of different search methods.
    
    Parameters:
    - query_results: List of dictionaries with timing results
    """
    # Extract query times
    queries = range(1, len(query_results) + 1)
    kd_times = [r['kd_query_time'] for r in query_results]
    one_d_times = [r['one_d_query_time'] for r in query_results]
    bf_times = [r['brute_force_time'] for r in query_results]
    match_counts = [r['brute_force_matches'] for r in query_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Query time comparison
    ax1.plot(queries, kd_times, marker='o', label='KD-Tree')
    ax1.plot(queries, one_d_times, marker='s', label='One-D Index')
    ax1.plot(queries, bf_times, marker='^', label='Brute Force')
    ax1.set_xlabel('Query Number')
    ax1.set_ylabel('Query Time (seconds)')
    ax1.set_title('Query Time Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Speedup comparison
    kd_speedups = [bf / kd for bf, kd in zip(bf_times, kd_times)]
    one_d_speedups = [bf / one_d for bf, one_d in zip(bf_times, one_d_times)]
    
    ax2.plot(queries, kd_speedups, marker='o', label='KD-Tree')
    ax2.plot(queries, one_d_speedups, marker='s', label='One-D Index')
    ax2.axhline(y=1, color='r', linestyle='--', label='Brute Force Baseline')
    ax2.set_xlabel('Query Number')
    ax2.set_ylabel('Speedup Factor (vs Brute Force)')
    ax2.set_title('Speedup Comparison')
    ax2.legend()
    ax2.grid(True)
    
    for i, count in enumerate(match_counts):
        ax2.annotate(f"{count}", 
                    (queries[i], max(kd_speedups[i], one_d_speedups[i]) + 0.5),
                    ha='center',
                    size=8)
    
    plt.tight_layout()
    plt.savefig('search_performance_comparison.png')
    print("\nPerformance visualization saved as 'search_performance_comparison.png'")


if __name__ == "__main__":
    try:
        install_package("pandas")
        install_package("openpyxl")
        install_package("matplotlib")
        install_package("numpy")
    except Exception as e:
        print(f"Error installing packages: {e}")
        print("Please make sure you have pip installed and try again.")
        sys.exit(1)

    # Load TPC-H Lineitem data from Excel
    file_path = "mp1_dataset_100k.xlsx"  # Change this to either the 100k version or the 10k version
    print(f"Loading data from {file_path}...")
    
    df = pd.read_excel(file_path, usecols=["l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice"])
    print(f"Loaded {len(df)} records")

    # Print sample data and dataset statistics
    print("\nSample data points:")
    for i, row in df.head(5).iterrows():
        print(row.tolist())
    
    print("\nData summary statistics:")
    for i, col in enumerate(["l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice"]):
        min_val = df[col].min()
        max_val = df[col].max()
        median = df[col].median()
        p25 = df[col].quantile(0.25)
        p75 = df[col].quantile(0.75)
        print(f"{col}: min={min_val}, 25th={p25}, median={median}, 75th={p75}, max={max_val}")

    # Convert data to list of tuples
    points = list(df.itertuples(index=False, name=None))
    
    # List to store results of all queries
    all_query_results = []

    # Define range query bounds based on actual data distribution
    query_bounds = [
        (df['l_orderkey'].min(), df['l_orderkey'].quantile(0.3)),  # lower 30% of orderkeys
        (df['l_partkey'].min(), df['l_partkey'].quantile(0.3)),    # lower 30% of partkeys
        (df['l_suppkey'].min(), df['l_suppkey'].quantile(0.5)),    # lower 50% of suppkeys
        (df['l_quantity'].min(), df['l_quantity'].quantile(0.5)),  # lower 50% of quantities
        (df['l_extendedprice'].min(), df['l_extendedprice'].quantile(0.3))  # lower 30% of prices
    ]
    
    print(f"\n\n=========== QUERY 1 ===========")
    print(f"Executing range query with data-driven bounds: {query_bounds}")
    results = compare_search_methods(points, query_bounds)
    all_query_results.append(results)

    # Experiment with different queries
    queries = [
        # Query 2: middle range
        [
            (df['l_orderkey'].quantile(0.3), df['l_orderkey'].quantile(0.6)),  # middle range of orderkeys
            (df['l_partkey'].quantile(0.3), df['l_partkey'].quantile(0.6)),    # middle range of partkeys
            (df['l_suppkey'].quantile(0.25), df['l_suppkey'].quantile(0.75)),  # middle 50% of suppkeys
            (df['l_quantity'].quantile(0.25), df['l_quantity'].quantile(0.75)),# middle 50% of quantities
            (df['l_extendedprice'].quantile(0.3), df['l_extendedprice'].quantile(0.6))  # middle range of prices
        ],
        # Query 3: mixed range
        [
            (df['l_orderkey'].quantile(0.1), df['l_orderkey'].quantile(0.4)),  # lower range of orderkeys
            (df['l_partkey'].quantile(0.6), df['l_partkey'].quantile(0.9)),    # higher range of partkeys
            (df['l_suppkey'].quantile(0.4), df['l_suppkey'].quantile(0.8)),    # upper middle range of suppkeys
            (df['l_quantity'].quantile(0.1), df['l_quantity'].quantile(0.5)),  # lower half of quantities
            (df['l_extendedprice'].quantile(0.7), df['l_extendedprice'].quantile(0.95)) # higher range of prices
        ],
        # Query 4: highly selective
        [
            (df['l_orderkey'].quantile(0.85), df['l_orderkey'].quantile(0.95)),  # high range of orderkeys
            (df['l_partkey'].quantile(0.85), df['l_partkey'].quantile(0.95)),    # high range of partkeys
            (df['l_suppkey'].quantile(0.85), df['l_suppkey'].quantile(0.95)),    # high range of suppkeys
            (df['l_quantity'].quantile(0.05), df['l_quantity'].quantile(0.15)),  # low range of quantities
            (df['l_extendedprice'].quantile(0.85), df['l_extendedprice'].quantile(0.95)) # high range of prices
        ],
        # Query 5: mixed selectivity
        [
            (df['l_orderkey'].min(), df['l_orderkey'].max()),  # full range of orderkeys
            (df['l_partkey'].quantile(0.3), df['l_partkey'].quantile(0.7)),  # middle range of partkeys
            (df['l_suppkey'].min(), df['l_suppkey'].max()),  # full range of suppkeys
            (df['l_quantity'].quantile(0.4), df['l_quantity'].quantile(0.6)),  # narrow middle range of quantities
            (df['l_extendedprice'].quantile(0.4), df['l_extendedprice'].quantile(0.6))  # narrow middle range of prices
        ],
        # Query 6: extremely selective
        [
            (df['l_orderkey'].quantile(0.95), df['l_orderkey'].quantile(0.98)),  # very high range of orderkeys
            (df['l_partkey'].quantile(0.95), df['l_partkey'].quantile(0.98)),    # very high range of partkeys
            (df['l_suppkey'].quantile(0.95), df['l_suppkey'].quantile(0.98)),    # very high range of suppkeys
            (df['l_quantity'].quantile(0.95), df['l_quantity'].quantile(0.98)),  # very high range of quantities
            (df['l_extendedprice'].quantile(0.95), df['l_extendedprice'].quantile(0.98))  # very high range of prices
        ]
    ]
    
    for i, bounds in enumerate(queries):
        print(f"\n\n=========== QUERY {i+2} ===========")
        print(f"Executing range query with bounds: {bounds}")
        results = compare_search_methods(points, bounds)
        all_query_results.append(results)
    
    # Summarize all query performance results
    print("\n\n===== PERFORMANCE SUMMARY =====")
    print("Query | Records | KD-Tree Time | One-D Index Time | Brute Force Time | KD Speedup | One-D Speedup")
    print("-" * 100)
    
    for i, result in enumerate(all_query_results):
        kd_speedup = result['brute_force_time'] / result['kd_query_time'] if result['kd_query_time'] > 0 else float('inf')
        one_d_speedup = result['brute_force_time'] / result['one_d_query_time'] if result['one_d_query_time'] > 0 else float('inf')
        
        print(f"{i+1:<5} | {result['brute_force_matches']:<7} | " +
              f"{result['kd_query_time']:.4f}s      | " +
              f"{result['one_d_query_time']:.4f}s           | " +
              f"{result['brute_force_time']:.4f}s          | " +
              f"{kd_speedup:.2f}x      | " +
              f"{one_d_speedup:.2f}x")
    
    # Create visualizations
    visualize_results(all_query_results)
    
    print("\nAll queries executed successfully")
