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


def benchmark_queries(kd_tree, df, num_runs=50):
    """
    Run multiple random queries and calculate average performance metrics
    
    Parameters:
    - kd_tree: The KD-tree to test
    - df: DataFrame containing the original data for generating valid random queries
    - num_runs: Number of different random queries to run (default: 50)
    
    Returns:
    - Dictionary with performance statistics
    """
    kd_times = []
    bf_times = []
    result_counts = []
    
    print(f"Running {num_runs} different random queries for performance benchmarking...")
    
    # Column names for reference
    columns = ["l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice"]
    
    for i in range(num_runs):
        # Generate random query bounds for each run
        query_bounds = []
        
        for j, col in enumerate(columns):
            # Get min and max values for this column
            min_val = df[col].min()
            max_val = df[col].max()
            
            # Generate random lower and upper bounds
            # Ensure lower < upper and both are within the data range
            lower = min_val + (max_val - min_val) * random.random() * 0.7
            upper = lower + (max_val - lower) * random.random()
            
            query_bounds.append((lower, upper))
        
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
        
        # Record number of results found
        result_counts.append(len(kd_results))
        
        # Verify results match for each run
        if set(kd_results) != set(bf_results):
            print(f"WARNING: Results differ on query {i+1}!")
        
        # Print progress every 10 queries
        if (i+1) % 10 == 0:
            print(f"Completed {i+1}/{num_runs} queries...")
    
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
    print(f"Average results per query: {sum(result_counts)/num_runs:.1f} records")
    print(f"Result range: {min(result_counts)} to {max(result_counts)} records")
    
    return {
        'kd_times': kd_times,
        'bf_times': bf_times,
        'result_counts': result_counts,
        'avg_kd_time': avg_kd_time,
        'avg_bf_time': avg_bf_time,
        'avg_speedup': avg_speedup
    }


def visualize_benchmark_results(benchmark_results):
    """
    Create detailed visualizations of the benchmark results
    
    Parameters:
    - benchmark_results: Dictionary returned by benchmark_queries function
    """
    # Extract data
    kd_times = benchmark_results['kd_times']
    bf_times = benchmark_results['bf_times']
    result_counts = benchmark_results['result_counts']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Query time comparison boxplot
    ax1 = fig.add_subplot(221)
    box_data = [kd_times, bf_times]
    box_labels = ['KD-Tree', 'Brute Force']
    ax1.boxplot(box_data, labels=box_labels)
    ax1.set_ylabel('Query Time (seconds)')
    ax1.set_title('Query Time Distribution')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add average lines
    for i, data in enumerate(box_data):
        avg = sum(data) / len(data)
        ax1.axhline(y=avg, color='red', linestyle='--', xmin=i/len(box_data), xmax=(i+1)/len(box_data))
        ax1.text(i+1, avg*1.05, f'Avg: {avg:.6f}s', horizontalalignment='center')
    
    # 2. Speedup vs Result Size scatter plot
    ax2 = fig.add_subplot(222)
    speedups = [bf / kd for bf, kd in zip(bf_times, kd_times)]
    ax2.scatter(result_counts, speedups, alpha=0.7)
    ax2.set_xlabel('Number of Results')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Speedup vs Result Size')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line
    z = np.polyfit(result_counts, speedups, 1)
    p = np.poly1d(z)
    ax2.plot(sorted(result_counts), p(sorted(result_counts)), "r--")
    
    # 3. Query time vs Result Size
    ax3 = fig.add_subplot(223)
    ax3.scatter(result_counts, kd_times, label='KD-Tree', alpha=0.7, color='blue')
    ax3.scatter(result_counts, bf_times, label='Brute Force', alpha=0.7, color='orange')
    ax3.set_xlabel('Number of Results')
    ax3.set_ylabel('Query Time (seconds)')
    ax3.set_title('Query Time vs Result Size')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Speedup Distribution Histogram
    ax4 = fig.add_subplot(224)
    ax4.hist(speedups, bins=20, alpha=0.7, color='green')
    ax4.set_xlabel('Speedup Factor')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Speedup Distribution')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Add average speedup line
    avg_speedup = sum(speedups) / len(speedups)
    ax4.axvline(x=avg_speedup, color='red', linestyle='--')
    ax4.text(avg_speedup*1.05, ax4.get_ylim()[1]*0.9, f'Avg: {avg_speedup:.2f}x', 
             horizontalalignment='left', verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('random_query_benchmark.png')
    print("Saved benchmark visualization to 'random_query_benchmark.png'")
    plt.show()


def visualize_selectivity_impact(benchmark_results, total_points):
    """
    Create a visualization showing the impact of query selectivity
    
    Parameters:
    - benchmark_results: Dictionary returned by benchmark_queries function
    - total_points: Total number of points in the dataset
    """
    # Extract data
    kd_times = benchmark_results['kd_times']
    bf_times = benchmark_results['bf_times']
    result_counts = benchmark_results['result_counts']
    
    # Calculate selectivity (percentage of data returned)
    selectivity = [count / total_points * 100 for count in result_counts]
    
    # Calculate speedup for each query
    speedups = [bf / kd for bf, kd in zip(bf_times, kd_times)]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with point size proportional to query time
    sizes = np.array(bf_times) * 5000  # Scale for visibility
    
    plt.scatter(selectivity, speedups, s=sizes, alpha=0.5, c=kd_times, cmap='viridis')
    plt.colorbar(label='KD-Tree Query Time (seconds)')
    
    plt.xlabel('Query Selectivity (% of data returned)')
    plt.ylabel('Speedup Factor')
    plt.title('Impact of Query Selectivity on KD-Tree Performance')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a few annotations for interesting points
    # Find indices of min and max speedup
    min_speedup_idx = speedups.index(min(speedups))
    max_speedup_idx = speedups.index(max(speedups))
    
    plt.annotate(f"{speedups[min_speedup_idx]:.2f}x",
                 (selectivity[min_speedup_idx], speedups[min_speedup_idx]),
                 xytext=(10, -20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'))
    
    plt.annotate(f"{speedups[max_speedup_idx]:.2f}x",
                 (selectivity[max_speedup_idx], speedups[max_speedup_idx]),
                 xytext=(10, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'))
    
    # Add trend line
    z = np.polyfit(selectivity, speedups, 2)  # Quadratic fit
    p = np.poly1d(z)
    x_line = np.linspace(min(selectivity), max(selectivity), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('selectivity_impact.png')
    print("Saved selectivity impact visualization to 'selectivity_impact.png'")
    plt.show()


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
    file_path = "mp1_dataset_100k.xlsx"  # Change this to your file's actual path
    print(f"Loading data from {file_path}...")
    
    try:
        df = pd.read_excel(file_path, usecols=["l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice"])
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please make sure the file exists and has the required columns.")
        sys.exit(1)

    # Convert to list of tuples
    points = list(df.itertuples(index=False, name=None))
    
    # Build KD-Tree
    print("\nBuilding KD-Tree...")
    start_time = time.time()
    kd_tree = KdTree(dim=5, points=points)
    build_time = time.time() - start_time
    print(f"KD-Tree built successfully in {build_time:.4f} seconds.")
    





    #done
    # Generate 50 unique queries with different bounds
    query_results = []
    for i in range(50):
        query_bounds = [
            (df[col].quantile(np.random.uniform(0, 0.4)), df[col].quantile(np.random.uniform(0.6, 1)))
            for col in ["l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice"]
        ]
        
        # Perform range query using KD-Tree
        start_time = time.time()
        kd_results = kd_tree.get_points_within_bounds(query_bounds)
        kd_query_time = time.time() - start_time
        
        # Perform brute-force search for comparison
        start_time = time.time()
        brute_force_results = kd_tree.brute_force_search(query_bounds)
        brute_force_time = time.time() - start_time
        
        # Store results
        query_results.append({
            "query_index": i + 1,
            "kd_query_time": kd_query_time,
            "brute_force_time": brute_force_time,
            "kd_results_count": len(kd_results),
            "brute_force_results_count": len(brute_force_results)
        })
        
        print(f"Query {i + 1}: KD-Tree={kd_query_time:.4f}s, Brute Force={brute_force_time:.4f}s, Matches={len(kd_results)}")
    



    #done
    # Analyze performance
    avg_kd_time = np.mean([q["kd_query_time"] for q in query_results])
    avg_brute_time = np.mean([q["brute_force_time"] for q in query_results])
    speedup_factor = avg_brute_time / avg_kd_time if avg_kd_time > 0 else float('inf')
    
    print("\nPerformance Summary:")
    print(f"- Average KD-Tree Query Time: {avg_kd_time:.4f} seconds")
    print(f"- Average Brute Force Query Time: {avg_brute_time:.4f} seconds")
    print(f"- Average Speedup Factor: {speedup_factor:.2f}x")
    
    # Visualize query performance
    plt.figure(figsize=(10, 5))
    plt.plot([q["query_index"] for q in query_results], [q["kd_query_time"] for q in query_results], label="KD-Tree", marker="o")
    plt.plot([q["query_index"] for q in query_results], [q["brute_force_time"] for q in query_results], label="Brute Force", marker="x")
    plt.xlabel("Query Index")
    plt.ylabel("Execution Time (s)")
    plt.title("KD-Tree vs Brute Force Query Performance")
    plt.legend()
    plt.grid()
    plt.show()
    
    print("\nAll queries executed and performance visualized successfully!")
