import subprocess
import sys
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


class KdTree:
    def __init__(self, dim, points):
        self._dim = dim
        self._points = points.copy()  # Create a copy to avoid modifying the original
        random.shuffle(self._points)  # Shuffle the dataset to ensure balanced splits
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

    def print_kd_tree(self, node=None, depth=0, max_depth=3):
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


def benchmark_queries(kd_tree, query_bounds, num_runs=50):
    """
    Run multiple queries and calculate average performance metrics
    
    Parameters:
    - kd_tree: The KD-tree to test
    - query_bounds: The query bounds to use
    - num_runs: Number of times to run each query (default: 50)
    
    Returns:
    - Dictionary with performance statistics
    """
    kd_times = []
    bf_times = []
    
    print(f"Running {num_runs} iterations for performance benchmarking...")
    
    for i in range(num_runs):
        # KD-Tree query
        start_time = time.time()
        kd_results = kd_tree.get_points_within_bounds(query_bounds)
        kd_time = time.time() - start_time
        kd_times.append(kd_time)
        
        # Brute force query
        start_time = time.time()
        bf_results = kd_tree.brute_force_search(query_bounds)
        bf_time = time.time() - start_time
        bf_times.append(bf_time)
        
        # Verify results match for each run (optional)
        if set(kd_results) != set(bf_results):
            print(f"WARNING: Results differ on run {i+1}!")
    
    # Calculate statistics
    avg_kd_time = sum(kd_times) / num_runs
    avg_bf_time = sum(bf_times) / num_runs
    min_kd_time = min(kd_times)
    max_kd_time = max(kd_times)
    min_bf_time = min(bf_times)
    max_bf_time = max(bf_times)
    avg_speedup = avg_bf_time / avg_kd_time if avg_kd_time > 0 else float('inf')
    
    # Results
    print("\nBenchmark Results:")
    print(f"KD-Tree query time: avg={avg_kd_time:.6f}s, min={min_kd_time:.6f}s, max={max_kd_time:.6f}s")
    print(f"Brute force time: avg={avg_bf_time:.6f}s, min={min_bf_time:.6f}s, max={max_bf_time:.6f}s")
    print(f"Average speedup factor: {avg_speedup:.2f}x")
    
    return {
        'kd_times': kd_times,
        'bf_times': bf_times,
        'avg_kd_time': avg_kd_time,
        'avg_bf_time': avg_bf_time,
        'avg_speedup': avg_speedup
    }


def visualize_kdtree_structure(kd_tree):
    """
    Create a visualization of the KD-tree structure
    """
    def count_nodes_at_depths(node, depth, counts):
        if node is None:
            return
        
        if depth >= len(counts):
            counts.append(0)
        
        counts[depth] += 1
        
        count_nodes_at_depths(node[0], depth + 1, counts)
        count_nodes_at_depths(node[1], depth + 1, counts)
    
    # Count nodes at each depth
    depth_counts = []
    count_nodes_at_depths(kd_tree._root, 0, depth_counts)
    
    # Plot tree structure
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(depth_counts)), depth_counts, color='skyblue')
    plt.xlabel('Tree Depth')
    plt.ylabel('Number of Nodes')
    plt.title('KD-Tree Structure: Node Distribution by Depth')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text annotations
    for i, count in enumerate(depth_counts):
        plt.text(i, count + max(depth_counts)*0.01, str(count), 
                 horizontalalignment='center')
    
    plt.tight_layout()
    plt.savefig('kdtree_structure.png')
    print("Saved node distribution visualization to 'kdtree_structure.png'")
    plt.show()
    
    # Additionally, create a visualization of the tree's shape
    max_depth = len(depth_counts) - 1
    
    # Create a nested list visualization
    def visualize_tree_shape(node, depth, x_min, x_max, result_array):
        if node is None:
            return
        
        # Calculate horizontal position (center of current range)
        x_pos = (x_min + x_max) / 2
        
        # Add node to result
        if depth < len(result_array):
            # Calculate a color based on which dimension is split
            axis = depth % kd_tree._dim
            result_array[depth][int(x_pos)] = axis + 1
        
        # Recurse on children
        mid = (x_min + x_max) / 2
        visualize_tree_shape(node[0], depth + 1, x_min, mid, result_array)
        visualize_tree_shape(node[1], depth + 1, mid, x_max, result_array)
    
    # Create a 2D array to represent the tree
    width = 2**(max_depth)  # Width needed for the bottom level
    tree_array = np.zeros((max_depth + 1, width))
    
    # Fill the array
    visualize_tree_shape(kd_tree._root, 0, 0, width, tree_array)
    
    # Create a colormap for dimensions
    cmap = colors.ListedColormap(['white', 'lightblue', 'skyblue', 'royalblue', 'navy', 'darkblue'])
    bounds = [0, 1, 2, 3, 4, 5, 6]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot the tree shape
    plt.figure(figsize=(12, 8))
    plt.imshow(tree_array, cmap=cmap, norm=norm, aspect='auto', interpolation='none')
    plt.colorbar(ticks=np.arange(0.5, 6), label='Dimension Index')
    plt.gca().set_yticks(np.arange(max_depth + 1))
    plt.gca().set_yticklabels(np.arange(max_depth + 1))
    plt.xlabel('Horizontal Position')
    plt.ylabel('Tree Depth')
    plt.title('KD-Tree Structure: Dimension Splits by Depth and Position')
    plt.tight_layout()
    plt.savefig('kdtree_dimensions.png')
    print("Saved dimension splits visualization to 'kdtree_dimensions.png'")
    plt.show()


def visualize_kdtree_2d_projection(points, kd_tree, dimensions=(0, 1)):
    """
    Create a 2D projection visualization of the KD-tree
    
    Parameters:
    - points: List of data points
    - kd_tree: KD-tree object
    - dimensions: Tuple of two dimensions to use for projection (default: first two dimensions)
    """
    # Extract the chosen dimensions from points
    x_dim, y_dim = dimensions
    x_values = [p[x_dim] for p in points]
    y_values = [p[y_dim] for p in points]
    
    # Create a recursive function to draw the tree divisions
    def draw_kdtree_divisions(node, depth, xmin, xmax, ymin, ymax):
        if node is None:
            return
            
        # Get the split dimension and value
        axis = depth % kd_tree._dim
        split_value = node[2][axis]
        
        # If the axis is one of our visualization dimensions, draw a line
        if axis == x_dim:
            plt.plot([split_value, split_value], [ymin, ymax], 'r-', alpha=0.3)
            draw_kdtree_divisions(node[0], depth + 1, xmin, split_value, ymin, ymax)
            draw_kdtree_divisions(node[1], depth + 1, split_value, xmax, ymin, ymax)
        elif axis == y_dim:
            plt.plot([xmin, xmax], [split_value, split_value], 'g-', alpha=0.3)
            draw_kdtree_divisions(node[0], depth + 1, xmin, xmax, ymin, split_value)
            draw_kdtree_divisions(node[1], depth + 1, xmin, xmax, split_value, ymax)
        else:
            # If splitting on a dimension we're not visualizing, just recurse on both sides
            draw_kdtree_divisions(node[0], depth + 1, xmin, xmax, ymin, ymax)
            draw_kdtree_divisions(node[1], depth + 1, xmin, xmax, ymin, ymax)
    
    # Plot the data points
    plt.figure(figsize=(10, 10))
    plt.scatter(x_values, y_values, s=5, alpha=0.5)
    
    # Get dimension names
    dim_names = ["l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice"]
    
    # Draw the tree divisions
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    # Add some padding
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    
    draw_kdtree_divisions(kd_tree._root, 0, x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)
    
    plt.xlabel(dim_names[x_dim])
    plt.ylabel(dim_names[y_dim])
    plt.title(f'2D Projection of KD-Tree: {dim_names[x_dim]} vs {dim_names[y_dim]}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'kdtree_2d_projection_{dim_names[x_dim]}_{dim_names[y_dim]}.png')
    print(f"Saved 2D projection visualization to 'kdtree_2d_projection_{dim_names[x_dim]}_{dim_names[y_dim]}.png'")
    plt.show()


def plot_comparison_chart(first_query_results, second_query_results):
    """
    Create a comparison chart of the two queries
    """
    # Extract data
    labels = ['First Query', 'Second Query']
    kd_times = [first_query_results['avg_kd_time'], second_query_results['avg_kd_time']]
    bf_times = [first_query_results['avg_bf_time'], second_query_results['avg_bf_time']]
    speedups = [first_query_results['avg_speedup'], second_query_results['avg_speedup']]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot average times
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, kd_times, width, label='KD-Tree')
    ax1.bar(x + width/2, bf_times, width, label='Brute Force')
    
    ax1.set_ylabel('Average Query Time (seconds)')
    ax1.set_title('Average Query Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    
    # Add text labels
    for i, v in enumerate(kd_times):
        ax1.text(i - width/2, v + 0.0001, f'{v:.6f}s', ha='center')
    
    for i, v in enumerate(bf_times):
        ax1.text(i + width/2, v + 0.0001, f'{v:.6f}s', ha='center')
    
    # Plot speedups
    ax2.bar(x, speedups, width, color='green')
    ax2.set_ylabel('Speedup Factor (x times)')
    ax2.set_title('KD-Tree Speedup Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    
    # Add text labels
    for i, v in enumerate(speedups):
        ax2.text(i, v + 0.5, f'{v:.2f}x', ha='center')
    
    plt.tight_layout()
    plt.savefig('query_comparison.png')
    print("Saved query comparison chart to 'query_comparison.png'")
    plt.show()


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
    file_path = "mp1_dataset_10k.xlsx"  # Change this to your file's actual path
    print(f"Loading data from {file_path}...")
    
    try:
        df = pd.read_excel(file_path, usecols=["l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice"])
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please make sure the file exists and has the required columns.")
        sys.exit(1)

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

    # Convert to list of tuples
    points = list(df.itertuples(index=False, name=None))
    
    # Build KD-Tree
    print("\nBuilding KD-Tree...")
    start_time = time.time()
    kd_tree = KdTree(dim=5, points=points)
    build_time = time.time() - start_time
    print(f"KD-Tree built successfully in {build_time:.4f} seconds.")
    
    # Print KD-Tree structure (limited to first few levels)
    print("\nKD-Tree Structure (showing first 3 levels):")
    kd_tree.print_kd_tree(max_depth=3)
    
    # Print tree statistics
    tree_stats = kd_tree.tree_stats()
    print("\nKD-Tree Statistics:")
    print(f"- Total nodes: {tree_stats['total_nodes']}")
    print(f"- Maximum depth: {tree_stats['max_depth']}")
    print(f"- Internal nodes: {tree_stats['internal_nodes']}")
    print(f"- Leaf nodes: {tree_stats['leaf_nodes']}")
    print(f"- Average depth: {tree_stats['avg_depth']:.2f}")
    print(f"- Nodes per level: {dict(sorted(tree_stats['depth_counts'].items()))}")

    # Define range query bounds based on actual data distribution
    query_bounds = [
        (df['l_orderkey'].min(), df['l_orderkey'].quantile(0.3)),  # lower 30% of orderkeys
        (df['l_partkey'].min(), df['l_partkey'].quantile(0.3)),    # lower 30% of partkeys
        (df['l_suppkey'].min(), df['l_suppkey'].quantile(0.5)),    # lower 50% of suppkeys
        (df['l_quantity'].min(), df['l_quantity'].quantile(0.5)),  # lower 50% of quantities
        (df['l_extendedprice'].min(), df['l_extendedprice'].quantile(0.3))  # lower 30% of prices
    ]

    print(f"\nExecuting range query with data-driven bounds: {query_bounds}")
    
    # Run a single instance to show sample results
    # Perform range query using KD-Tree
    start_time = time.time()
    kd_results = kd_tree.get_points_within_bounds(query_bounds)
    kd_query_time = time.time() - start_time
    print(f"KD-Tree query completed in {kd_query_time:.4f} seconds.")
    print(f"Found {len(kd_results)} matching records")

    # Sample output of first few results
    print("\nSample KD-Tree Results:")
    for row in kd_results[:5]:  # Show first 5 results
        print(row)
    if len(kd_results) > 5:
        print(f"... and {len(kd_results) - 5} more")

    # Perform brute-force search for comparison
    print("\nExecuting brute force search with the same bounds...")
    start_time = time.time()
    brute_force_results = kd_tree.brute_force_search(query_bounds)
    brute_force_time = time.time() - start_time
    print(f"Brute force search completed in {brute_force_time:.4f} seconds.")
    print(f"Found {len(brute_force_results)} matching records")

    # Verify that both methods give the same results
    kd_set = set(kd_results)
    bf_set = set(brute_force_results)
    
    if kd_set == bf_set:
        print("\nBoth methods returned identical results!")
    else:
        print("\nWARNING: Results differ between methods!")
        print(f"KD-Tree unique results: {len(kd_set)}")
        print(f"Brute Force unique results: {len(bf_set)}")
        print(f"Missing from KD-Tree: {len(bf_set - kd_set)}")
        print(f"Extra in KD-Tree: {len(kd_set - bf_set)}")
    
    # Define second query bounds
    query_bounds2 = [
        (df['l_orderkey'].quantile(0.3), df['l_orderkey'].quantile(0.6)),  # middle range of orderkeys
        (df['l_partkey'].quantile(0.3), df['l_partkey'].quantile(0.6)),    # middle range of partkeys
        (df['l_suppkey'].quantile(0.25), df['l_suppkey'].quantile(0.75)),  # middle 50% of suppkeys
        (df['l_quantity'].quantile(0.25), df['l_quantity'].quantile(0.75)),# middle 50% of quantities
        (df['l_extendedprice'].quantile(0.3), df['l_extendedprice'].quantile(0.6))  # middle range of prices
    ]
    
    print(f"\nExecuting second range query with bounds: {query_bounds2}")
    
    # Run a single instance to show sample results
    # Perform range query using KD-Tree
    start_time = time.time()
    kd_results2 = kd_tree.get_points_within_bounds(query_bounds2)
    kd_query_time2 = time.time() - start_time
    print(f"KD-Tree query completed in {kd_query_time2:.4f} seconds.")
    print(f"Found {len(kd_results2)} matching records")

    # Perform brute-force search for comparison
    start_time = time.time()
    brute_force_results2 = kd_tree.brute_force_search(query_bounds2)
    brute_force_time2 = time.time() - start_time
    print(f"Brute force search completed in {brute_force_time2:.4f} seconds.")
    print(f"Found {len(brute_force_results2)} matching records")
    
    # Verify that both methods give the same results
    kd_set2 = set(kd_results2)
    bf_set2 = set(brute_force_results2)
    
    if kd_set2 == bf_set2:
        print("\nBoth methods returned identical results!")
    else:
        print("\nWARNING: Results differ between methods!")
    
    # Performance comparison
    if brute_force_time2 > 0 and kd_query_time2 > 0:
        print(f"Speedup factor: {brute_force_time2 / kd_query_time2:.2f}x")
    
    # ==================== New code for benchmarking and visualization ===================
    
    # Run benchmarks on first query
    print("\n" + "="*50)
    print("BENCHMARKING AND VISUALIZATION")
    print("="*50)
    
    print("\nBenchmarking first query...")
    first_query_results = benchmark_queries(kd_tree, query_bounds)
    
    # Run benchmarks on second query
    print("\nBenchmarking second query...")
    second_query_results = benchmark_queries(kd_tree, query_bounds2)
    
    # Compare the two queries
    print("\nComparison between queries:")
    print(f"First query average speedup: {first_query_results['avg_speedup']:.2f}x")
    print(f"Second query average speedup: {second_query_results['avg_speedup']:.2f}x")
    
    # Plot comparison chart
    plot_comparison_chart(first_query_results, second_query_results)
    
    # Visualize the KD-tree structure
    print("\nGenerating KD-tree structure visualizations...")
    visualize_kdtree_structure(kd_tree)
    
    # Create 2D projections using different combinations of dimensions
    print("\nCreating 2D projections of the KD-tree...")
    
    # Orderkey vs Partkey
    visualize_kdtree_2d_projection(points, kd_tree, dimensions=(0, 1))
    
    # Quantity vs Price 
    visualize_kdtree_2d_projection(points, kd_tree, dimensions=(3, 4))
    
    # Suppkey vs Price (another interesting combination)
    visualize_kdtree_2d_projection(points, kd_tree, dimensions=(2, 4))
    
    print("\nAll visualizations and benchmarks completed successfully!")
