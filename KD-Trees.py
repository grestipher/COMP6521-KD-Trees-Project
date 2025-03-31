import subprocess
import sys
import pandas as pd
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


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
    file_path = "mp1_dataset_10k.xlsx"  # Change this to either the 100k version or the 10k version
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





    # Sample output of first few results
    print("\nSample Brute Force Results:")
    for row in brute_force_results[:5]:  # Show first 5 results
        print(row)
    if len(brute_force_results) > 5:
        print(f"... and {len(brute_force_results) - 5} more")

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
    
    # Performance comparison
    print("\nPerformance Comparison:")
    print(f"KD-Tree query time: {kd_query_time:.4f} seconds")
    print(f"Brute force time: {brute_force_time:.4f} seconds")
    if brute_force_time > 0 and kd_query_time > 0:
        print(f"Speedup factor: {brute_force_time / kd_query_time:.2f}x")
    
    # Try another query with different bounds for comparison
    print("\n\nTrying another query with different bounds...")
    query_bounds2 = [
        (df['l_orderkey'].quantile(0.3), df['l_orderkey'].quantile(0.6)),  # middle range of orderkeys
        (df['l_partkey'].quantile(0.3), df['l_partkey'].quantile(0.6)),    # middle range of partkeys
        (df['l_suppkey'].quantile(0.25), df['l_suppkey'].quantile(0.75)),  # middle 50% of suppkeys
        (df['l_quantity'].quantile(0.25), df['l_quantity'].quantile(0.75)),# middle 50% of quantities
        (df['l_extendedprice'].quantile(0.3), df['l_extendedprice'].quantile(0.6))  # middle range of prices
    ]
    
    print(f"\nExecuting second range query with bounds: {query_bounds2}")
    
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
    

    # Generate n unique queries with different bounds
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
    
    print("\nAll queries executed and plotted successfully")