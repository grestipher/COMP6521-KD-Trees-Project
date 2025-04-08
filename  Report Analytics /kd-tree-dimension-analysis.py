import subprocess
import sys
import pandas as pd
import random
import time
import numpy as np
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


class KdTree:
    def __init__(self, dim, points, dim_indices):
        self._dim = dim
        self._dim_indices = dim_indices  # Store which dimensions we're using
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
        points.sort(key=lambda x: x[self._dim_indices[axis]])
        
        median_idx = len(points) // 2
        
        return [
            self._make_kd_tree_rec(points[:median_idx], depth + 1),
            self._make_kd_tree_rec(points[median_idx + 1:], depth + 1),
            points[median_idx]
        ]

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
        """
        Find all points in the KD-tree that lie within the given bounds.
        
        Args:
            bounds: A list of (min, max) tuples for each dimension we're using.
                   It must have the same length as self._dim_indices.
        
        Returns:
            A list of points that lie within the bounds.
        """
        if not self._root:
            return []
        return self._get_points_within_bounds_rec(self._root, bounds)

    def _get_points_within_bounds_rec(self, kd_node, bounds, depth=0):
        if kd_node is None:
            return []
        
        results = []
        axis = depth % self._dim
        dim_idx = self._dim_indices[axis]
        
        # Check if the current point is within bounds
        if kd_node[2] is not None and self._is_point_within_bounds(kd_node[2], bounds):
            results.append(kd_node[2])
        
        # Check if the splitting plane intersects the query box
        # If so, we need to check both left and right subtrees
        if kd_node[0] is not None and (kd_node[2] is None or bounds[axis][0] <= kd_node[2][dim_idx]):
            results.extend(self._get_points_within_bounds_rec(kd_node[0], bounds, depth + 1))
        
        if kd_node[1] is not None and (kd_node[2] is None or bounds[axis][1] >= kd_node[2][dim_idx]):
            results.extend(self._get_points_within_bounds_rec(kd_node[1], bounds, depth + 1))
        
        return results

    def _is_point_within_bounds(self, point, bounds):
        """
        Check if a point is within the given bounds for the dimensions we care about.
        """
        return all(bounds[i][0] <= point[self._dim_indices[i]] <= bounds[i][1] for i in range(self._dim))

    # Brute-force search for comparison
    def brute_force_search(self, bounds):
        """
        Perform a brute-force search through all points to find those within bounds.
        
        Args:
            bounds: A list of (min, max) tuples for each dimension we're using.
        
        Returns:
            A list of points that lie within the bounds.
        """
        return [point for point in self._points if self._is_point_within_bounds(point, bounds)]


def run_dimension_combination_test(df, all_columns, combination_size=3, num_queries=5):
    """
    Test KD-tree performance on different combinations of dimensions.
    
    Args:
        df: DataFrame containing the data
        all_columns: List of all column names to consider
        combination_size: Size of dimension combinations to test
        num_queries: Number of random queries to run for each tree
        
    Returns:
        Dictionary with results for each combination
    """
    results = {}
    
    # Convert DataFrame to list of tuples (full rows)
    full_points = list(df.itertuples(index=False, name=None))
    
    print(f"\n--- Testing KD-trees with {combination_size} dimensions ---")
    
    # Get all combinations of the specified size
    for combo in itertools.combinations(range(len(all_columns)), combination_size):
        combo_name = '-'.join([all_columns[i] for i in combo])
        print(f"\nTesting combination: {combo_name}")
        
        # Build KD-tree with this combination of dimensions
        start_time = time.time()
        kd_tree = KdTree(dim=len(combo), points=full_points, dim_indices=combo)
        build_time = time.time() - start_time
        print(f"Tree built in {build_time:.4f} seconds")
        
        # Run several queries with different bounds
        kd_query_times = []
        bf_query_times = []
        result_counts = []
        speedups = []
        
        for q in range(num_queries):
            # Create random query bounds for the selected dimensions
            bounds = []
            for dim_idx in combo:
                col = all_columns[dim_idx]
                # Random bounds between 10th and 90th percentile for this query
                lower_pct = random.uniform(0.1, 0.6)
                upper_pct = random.uniform(lower_pct + 0.1, 0.9)
                lower = df[col].quantile(lower_pct)
                upper = df[col].quantile(upper_pct)
                bounds.append((lower, upper))
            
            # KD-tree query
            start_time = time.time()
            kd_results = kd_tree.get_points_within_bounds(bounds)
            kd_time = time.time() - start_time
            kd_query_times.append(kd_time)
            
            # Brute force query
            start_time = time.time()
            bf_results = kd_tree.brute_force_search(bounds)
            bf_time = time.time() - start_time
            bf_query_times.append(bf_time)
            
            # Calculate speedup
            if bf_time > 0 and kd_time > 0:
                speedup = bf_time / kd_time
            else:
                speedup = 1.0
            speedups.append(speedup)
            
            # Verify results match
            kd_set = set(kd_results)
            bf_set = set(bf_results)
            if kd_set != bf_set:
                print(f"WARNING: Results differ for query {q+1}")
            
            result_counts.append(len(kd_results))
            
            print(f"  Query {q+1}: Found {len(kd_results)} records, KD: {kd_time:.4f}s, BF: {bf_time:.4f}s, Speedup: {speedup:.2f}x")
        
        # Store results for this combination
        results[combo_name] = {
            'dimensions': combo,
            'build_time': build_time,
            'avg_kd_query_time': sum(kd_query_times) / len(kd_query_times),
            'avg_bf_query_time': sum(bf_query_times) / len(bf_query_times),
            'avg_speedup': sum(speedups) / len(speedups),
            'avg_result_count': sum(result_counts) / len(result_counts),
            'max_speedup': max(speedups),
            'min_speedup': min(speedups),
        }
        
        # Tree stats
        tree_stats = kd_tree.tree_stats()
        results[combo_name].update({
            'tree_depth': tree_stats['max_depth'],
            'avg_node_depth': tree_stats['avg_depth'],
            'total_nodes': tree_stats['total_nodes'],
        })
        
    return results


def test_all_combination_sizes(df, columns, max_combo_size=4, num_queries=3):
    """
    Test KD-trees with combinations of different sizes
    """
    all_results = {}
    
    # Test combinations of different sizes
    for size in range(2, max_combo_size + 1):
        size_results = run_dimension_combination_test(df, columns, size, num_queries)
        all_results.update(size_results)
    
    return all_results


def visualize_results(results):
    """Create visualizations of the results"""
    
    # Extract data for plotting
    combo_names = list(results.keys())
    build_times = [results[c]['build_time'] for c in combo_names]
    avg_speedups = [results[c]['avg_speedup'] for c in combo_names]
    tree_depths = [results[c]['tree_depth'] for c in combo_names]
    avg_result_counts = [results[c]['avg_result_count'] for c in combo_names]
    
    # Group results by dimension count
    dim_counts = defaultdict(list)
    for combo in combo_names:
        dim_count = len(combo.split('-'))
        dim_counts[dim_count].append(combo)
    
    # Create plots
    plt.figure(figsize=(12, 10))
    
    # 1. Build times vs dimension combination
    plt.subplot(2, 2, 1)
    plt.bar(combo_names, build_times)
    plt.xticks(rotation=90, fontsize=8)
    plt.title('KD-Tree Build Time by Dimension Combination')
    plt.ylabel('Build Time (seconds)')
    plt.tight_layout()

    # 2. Average speedup vs dimension combination
    plt.subplot(2, 2, 2)
    bars = plt.bar(combo_names, avg_speedups)
    plt.xticks(rotation=90, fontsize=8)
    plt.title('Average Speedup by Dimension Combination')
    plt.ylabel('Speedup Factor (BF/KD)')
    
    # Color bars by dimension count
    for dim_count, combos in dim_counts.items():
        color = plt.cm.viridis(dim_count / (max(dim_counts.keys()) + 1))
        for combo in combos:
            idx = combo_names.index(combo)
            bars[idx].set_color(color)
    plt.tight_layout()
    
    # 3. Tree depth vs dimension count
    plt.subplot(2, 2, 3)
    x_coords = list(range(len(combo_names)))
    dim_counts_per_combo = [len(combo.split('-')) for combo in combo_names]
    plt.scatter(dim_counts_per_combo, tree_depths)
    plt.title('Tree Depth vs Dimension Count')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Tree Depth')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 4. Average speedup vs average result count
    plt.subplot(2, 2, 4)
    scatter = plt.scatter(avg_result_counts, avg_speedups, 
                         c=[len(combo.split('-')) for combo in combo_names], 
                         cmap='viridis', alpha=0.7)
    plt.title('Speedup vs Result Set Size')
    plt.xlabel('Average Number of Results')
    plt.ylabel('Average Speedup Factor')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.colorbar(scatter, label='Dimension Count')
    plt.tight_layout()
    
    plt.savefig('kd_tree_performance_results.png')
    plt.close()
    
    # Also create a summary table sorted by speedup
    summary_data = []
    for combo in combo_names:
        summary_data.append({
            'Dimensions': combo,
            'Dim Count': len(combo.split('-')),
            'Build Time': results[combo]['build_time'],
            'Avg Speedup': results[combo]['avg_speedup'],
            'Avg Results': results[combo]['avg_result_count'],
            'Tree Depth': results[combo]['tree_depth']
        })
    
    # Sort by average speedup
    summary_df = pd.DataFrame(summary_data).sort_values('Avg Speedup', ascending=False)
    summary_df.to_csv('kd_tree_performance_summary.csv', index=False)
    
    return summary_df


if __name__ == "__main__":
    try:
        # Install required packages
        for package in ["pandas", "openpyxl", "matplotlib", "numpy"]:
            try:
                install_package(package)
            except Exception as e:
                print(f"Error installing {package}: {e}")
                print("Please install the package manually and try again.")
    except Exception as e:
        print(f"Error installing packages: {e}")
        print("Please make sure you have pip installed and try again.")
        sys.exit(1)

    print("Loading TPC-H lineitem data...")
    try:
        file_path = "mp1_dataset_10k.xlsx"  # Change to your dataset path
        df = pd.read_excel(file_path)
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please make sure the file exists and is in Excel format.")
        sys.exit(1)
    
    # Check if all the required columns exist
    required_columns = [
        "l_orderkey", "l_partkey", "l_suppkey", "l_linenumber",
        "l_quantity", "l_extendedprice", "l_discount", "l_tax"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        # Use only available columns
        columns = [col for col in required_columns if col in df.columns]
        print(f"Proceeding with available columns: {columns}")
    else:
        columns = required_columns
    
    print(f"\nPerforming tests with columns: {columns}")
    print("This may take some time depending on the dataset size...")
    
    # Run the tests with different combination sizes
    max_combo_size = min(4, len(columns))  # Limit to avoid too many combinations
    results = test_all_combination_sizes(df, columns, max_combo_size=max_combo_size, num_queries=3)
    
    # Create visualizations and summary
    summary = visualize_results(results)
    
    # Print the top 5 best combinations
    print("\n===== TOP 5 DIMENSION COMBINATIONS BY SPEEDUP =====")
    print(summary.head(5).to_string(index=False))
    
    # Print the bottom 5 combinations
    print("\n===== BOTTOM 5 DIMENSION COMBINATIONS BY SPEEDUP =====")
    print(summary.tail(5).to_string(index=False))
    
    print("\nAnalysis complete! Results saved to kd_tree_performance_summary.csv and visualization to kd_tree_performance_results.png")
