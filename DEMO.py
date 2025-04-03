import subprocess
import sys
import pandas as pd
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import datetime


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


class KdTree:
    def __init__(self, columns, points, column_names=None):
        """
        Initialize KD-Tree with specified columns only
        
        Args:
            columns: List of column indices to use for the KD-Tree
            points: List of data points (tuples)
            column_names: Optional list of column names for display
        """
        self._columns = columns
        self._dim = len(columns)
        
        # Extract only the specified columns from each point
        self._points = []
        for point in points:
            # Create new point with only the specified columns
            new_point = tuple(point[col] for col in columns)
            # Store original point as metadata in the last position
            self._points.append(new_point + (point,))
        
        self._column_names = column_names if column_names else [f"Column {i}" for i in columns]
        
        # Shuffle points for better balance
        random.shuffle(self._points)  
        self._build_time = time.time()
        self._root = self._make_kd_tree()
        self._build_time = time.time() - self._build_time

    def _make_kd_tree(self):
        if not self._points:
            return None
        return self._make_kd_tree_rec(self._points, 0)

    def _make_kd_tree_rec(self, points, depth=0):
        if not points:
            return None
        
        axis = depth % self._dim
        
        # Sort based on the current axis
        points.sort(key=lambda x: x[axis])
        
        median_idx = len(points) // 2
        
        return [
            self._make_kd_tree_rec(points[:median_idx], depth + 1),
            self._make_kd_tree_rec(points[median_idx + 1:], depth + 1),
            points[median_idx]
        ]

    def print_kd_tree(self, node=None, depth=0, max_depth=5):
        """Print a visual representation of the KD-tree structure."""
        if node is None:
            node = self._root
        
        if node is None:
            return
            
        indent = "  " * depth
        
        if depth <= max_depth:
            axis = depth % self._dim
            axis_name = self._column_names[axis]
            print(f"{indent}Depth {depth} (split on {axis_name}): {node[2][:-1]}")  # Exclude original point
            
            if depth == max_depth:
                print(f"{indent}  ... (tree continues)")
            else:
                self.print_kd_tree(node[0], depth + 1, max_depth)
                self.print_kd_tree(node[1], depth + 1, max_depth)

    def get_build_time(self):
        """Return the time taken to build the KD-Tree"""
        return self._build_time

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
        Range query: Find all points within specified bounds
        
        Args:
            bounds: List of (min, max) tuples for each dimension used in the tree
        
        Returns:
            List of original data points that satisfy the bounds
        """
        if not self._root:
            return []
        results = self._get_points_within_bounds_rec(self._root, bounds)
        # Return the original points (stored as metadata)
        return [result[-1] for result in results]

    def _get_points_within_bounds_rec(self, kd_node, bounds, depth=0):
        if kd_node is None:
            return []
        
        results = []
        axis = depth % self._dim
        
        # Check if the current point is within bounds
        if kd_node[2] is not None and self._is_point_within_bounds(kd_node[2], bounds):
            results.append(kd_node[2])
        
        # Check which subtrees to traverse
        if kd_node[0] is not None and (kd_node[2] is None or bounds[axis][0] <= kd_node[2][axis]):
            results.extend(self._get_points_within_bounds_rec(kd_node[0], bounds, depth + 1))
        
        if kd_node[1] is not None and (kd_node[2] is None or bounds[axis][1] >= kd_node[2][axis]):
            results.extend(self._get_points_within_bounds_rec(kd_node[1], bounds, depth + 1))
        
        return results

    def get_exact_matches(self, match_values):
        """
        Match query: Find all points that exactly match the specified values
        
        Args:
            match_values: List of values to match for each dimension used in the tree,
                          None for dimensions that should be ignored
        
        Returns:
            List of original data points that exactly match the criteria
        """
        if not self._root:
            return []
        results = self._get_exact_matches_rec(self._root, match_values)
        # Return the original points (stored as metadata)
        return [result[-1] for result in results]

    def _get_exact_matches_rec(self, kd_node, match_values, depth=0):
        if kd_node is None:
            return []
        
        results = []
        axis = depth % self._dim
        
        # Check if current point matches
        if kd_node[2] is not None and self._is_exact_match(kd_node[2], match_values):
            results.append(kd_node[2])
        
        # If we don't care about this dimension or current value matches, check both subtrees
        if match_values[axis] is None:
            results.extend(self._get_exact_matches_rec(kd_node[0], match_values, depth + 1))
            results.extend(self._get_exact_matches_rec(kd_node[1], match_values, depth + 1))
        else:
            # Current dimension matters for matching
            if kd_node[2] is not None:
                # Search left subtree if match value is less than current node's value
                if match_values[axis] < kd_node[2][axis]:
                    results.extend(self._get_exact_matches_rec(kd_node[0], match_values, depth + 1))
                # Search right subtree if match value is greater than current node's value
                elif match_values[axis] > kd_node[2][axis]:
                    results.extend(self._get_exact_matches_rec(kd_node[1], match_values, depth + 1))
                # If values are equal, only need to search both if we're not at a leaf
                else:
                    results.extend(self._get_exact_matches_rec(kd_node[0], match_values, depth + 1))
                    results.extend(self._get_exact_matches_rec(kd_node[1], match_values, depth + 1))
                    
        return results

    def _is_exact_match(self, point, match_values):
        """Check if a point exactly matches the specified values"""
        for i in range(len(match_values)):
            if match_values[i] is not None and point[i] != match_values[i]:
                return False
        return True

    def _is_point_within_bounds(self, point, bounds):
        """Check if a point falls within the specified bounds"""
        for i in range(self._dim):
            if point[i] < bounds[i][0] or point[i] > bounds[i][1]:
                return False
        return True

    # Brute force search for comparison
    def brute_force_range_search(self, bounds):
        """Brute force implementation of range query for comparison"""
        results = []
        for point in self._points:
            if self._is_point_within_bounds(point, bounds):
                results.append(point[-1])  # Return original point
        return results

    def brute_force_exact_match(self, match_values):
        """Brute force implementation of exact match query for comparison"""
        results = []
        for point in self._points:
            if self._is_exact_match(point, match_values):
                results.append(point[-1])  # Return original point
        return results


# Function to convert datetime to numeric for KD-Tree
def datetime_to_numeric(dt):
    """Convert datetime to numeric value for use in KD-Tree"""
    if isinstance(dt, datetime.datetime) or isinstance(dt, datetime.date):
        # Convert to days since epoch
        return (dt - datetime.datetime(1970, 1, 1).date()).days
    return dt


# Function to build KD-Tree for specified columns
def build_kd_tree_for_columns(df, column_indices):
    """
    Build a KD-Tree using only the specified columns
    
    Args:
        df: Pandas DataFrame with the data
        column_indices: List of column indices to include in the KD-Tree
    
    Returns:
        KdTree object built with the specified columns
    """
    # Convert DataFrame to list of tuples
    columns = df.columns.tolist()
    points = []
    
    for row in df.itertuples(index=False, name=None):
        # Convert any datetime columns to numeric
        processed_row = tuple(datetime_to_numeric(val) for val in row)
        points.append(processed_row)
    
    # Build KD-Tree with only the specified columns
    column_names = [columns[i] for i in column_indices]
    print(f"Building KD-Tree for columns: {column_names}")
    
    start_time = time.time()
    kd_tree = KdTree(column_indices, points, column_names)
    build_time = time.time() - start_time
    
    print(f"KD-Tree built in {build_time:.4f} seconds")
    
    # Print tree statistics
    tree_stats = kd_tree.tree_stats()
    print("\nKD-Tree Statistics:")
    print(f"- Total nodes: {tree_stats['total_nodes']}")
    print(f"- Maximum depth: {tree_stats['max_depth']}")
    print(f"- Internal nodes: {tree_stats['internal_nodes']}")
    print(f"- Leaf nodes: {tree_stats['leaf_nodes']}")
    print(f"- Average depth: {tree_stats['avg_depth']:.2f}")
    
    return kd_tree


def run_range_query_demo(df, column_indices):
    """
    Run a range query demo on the specified columns
    
    Args:
        df: Pandas DataFrame with the data
        column_indices: List of column indices to use for the KD-Tree
    """
    # Build KD-Tree for specified columns
    kd_tree = build_kd_tree_for_columns(df, column_indices)
    
    # Define query bounds for selected columns
    bounds = []
    for idx in column_indices:
        col = df.columns[idx]
        # Handle datetime columns
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            min_val = datetime_to_numeric(df[col].min())
            max_val = datetime_to_numeric(df[col].quantile(0.5))
        else:
            min_val = df[col].min()
            max_val = df[col].quantile(0.5)
        bounds.append((min_val, max_val))
    
    print(f"\nExecuting range query with bounds:")
    for i, idx in enumerate(column_indices):
        col = df.columns[idx]
        print(f"- {col}: {bounds[i]}")
        
    # Execute KD-Tree query
    start_time = time.time()
    kd_results = kd_tree.get_points_within_bounds(bounds)
    kd_query_time = time.time() - start_time
    print(f"\nKD-Tree query completed in {kd_query_time:.4f} seconds.")
    print(f"Found {len(kd_results)} matching records")
    
    # Execute brute force query
    start_time = time.time()
    brute_force_results = kd_tree.brute_force_range_search(bounds)
    brute_force_time = time.time() - start_time
    print(f"Brute force search completed in {brute_force_time:.4f} seconds.")
    print(f"Found {len(brute_force_results)} matching records")
    
    # Compare results
    if len(kd_results) == len(brute_force_results):
        print("\nBoth methods returned the same number of results.")
        if set(kd_results) == set(brute_force_results):
            print("Results are identical!")
        else:
            print("WARNING: Results differ!")
    else:
        print(f"\nWARNING: Result counts differ! KD-Tree: {len(kd_results)}, Brute Force: {len(brute_force_results)}")
    
    # Performance comparison
    if brute_force_time > 0 and kd_query_time > 0:
        print(f"Speedup factor: {brute_force_time / kd_query_time:.2f}x")
    
    # Show sample results
    print("\nSample results:")
    for row in kd_results[:5]:
        print(row)
    if len(kd_results) > 5:
        print(f"... and {len(kd_results) - 5} more")

    # Plot performance comparison
    labels = ['KD-Tree', 'Brute Force']
    times = [kd_query_time, brute_force_time]
    
    plt.figure(figsize=(8, 4))
    plt.bar(labels, times, color=['blue', 'orange'])
    plt.ylabel('Time (seconds)')
    plt.title('Query Performance Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(times):
        plt.text(i, v + 0.01, f"{v:.4f}s", ha='center')
    plt.tight_layout()
    plt.show()


def run_exact_match_demo(df, column_indices):
    """
    Run an exact match query demo on the specified columns
    
    Args:
        df: Pandas DataFrame with the data
        column_indices: List of column indices to use for the KD-Tree
    """
    # Build KD-Tree for specified columns
    kd_tree = build_kd_tree_for_columns(df, column_indices)
    
    # Choose values for exact match query
    # For each column, either select a specific value or None to ignore
    match_values = []
    for idx in column_indices:
        col = df.columns[idx]
        # For this demo, use median value for half of columns, None for others
        if random.random() > 0.5:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                val = datetime_to_numeric(df[col].sample(n=1).iloc[0])
            else:
                val = df[col].sample(n=1).iloc[0]
            match_values.append(val)
        else:
            match_values.append(None)
    
    print(f"\nExecuting exact match query with values:")
    for i, idx in enumerate(column_indices):
        col = df.columns[idx]
        value = match_values[i]
        if value is None:
            print(f"- {col}: Any value")
        else:
            print(f"- {col}: {value}")
    
    # Execute KD-Tree query
    start_time = time.time()
    kd_results = kd_tree.get_exact_matches(match_values)
    kd_query_time = time.time() - start_time
    print(f"\nKD-Tree query completed in {kd_query_time:.4f} seconds.")
    print(f"Found {len(kd_results)} matching records")
    
    # Execute brute force query
    start_time = time.time()
    brute_force_results = kd_tree.brute_force_exact_match(match_values)
    brute_force_time = time.time() - start_time
    print(f"Brute force search completed in {brute_force_time:.4f} seconds.")
    print(f"Found {len(brute_force_results)} matching records")
    
    # Compare results
    if len(kd_results) == len(brute_force_results):
        print("\nBoth methods returned the same number of results.")
        if set(kd_results) == set(brute_force_results):
            print("Results are identical!")
        else:
            print("WARNING: Results differ!")
    else:
        print(f"\nWARNING: Result counts differ! KD-Tree: {len(kd_results)}, Brute Force: {len(brute_force_results)}")
    
    # Performance comparison
    if brute_force_time > 0 and kd_query_time > 0:
        print(f"Speedup factor: {brute_force_time / kd_query_time:.2f}x")
    
    # Show sample results
    print("\nSample results:")
    for row in kd_results[:5]:
        print(row)
    if len(kd_results) > 5:
        print(f"... and {len(kd_results) - 5} more")
    
    # Plot performance comparison
    labels = ['KD-Tree', 'Brute Force']
    times = [kd_query_time, brute_force_time]
    
    plt.figure(figsize=(8, 4))
    plt.bar(labels, times, color=['blue', 'orange'])
    plt.ylabel('Time (seconds)')
    plt.title('Exact Match Query Performance Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(times):
        plt.text(i, v + 0.01, f"{v:.4f}s", ha='center')
    plt.tight_layout()
    plt.show()


def run_custom_query(df):
    """
    Allow user to interactively select columns and query type
    
    Args:
        df: Pandas DataFrame with the data
    """
    # Display available columns
    print("\n=== Available Columns ===")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")
    
    # Get columns from user
    try:
        col_input = input("\nEnter column indices to use (comma-separated): ")
        column_indices = [int(idx.strip()) for idx in col_input.split(',')]
        
        # Validate column indices
        for idx in column_indices:
            if idx < 0 or idx >= len(df.columns):
                print(f"Invalid column index: {idx}")
                return
        
        # Get query type
        query_type = input("\nEnter query type (range/match): ").strip().lower()
        
        if query_type == 'range':
            run_range_query_demo(df, column_indices)
        elif query_type == 'match':
            run_exact_match_demo(df, column_indices)
        else:
            print(f"Invalid query type: {query_type}. Please use 'range' or 'match'.")
    except ValueError:
        print("Invalid input. Please enter column indices as comma-separated integers.")
    except Exception as e:
        print(f"Error: {e}")


def run_flexible_query_demo(df):
    """Main demo function for flexible queries with user input values"""
    # Map column names to indices
    columns = {name: i for i, name in enumerate(df.columns)}
    
    print("\n=== Available Columns ===")
    for name, idx in columns.items():
        print(f"{idx}: {name}")
    
    # Numeric columns (excluding dates)
    numeric_columns = [i for i, col in enumerate(df.columns) if 
                      pd.api.types.is_numeric_dtype(df[col]) and 
                      not pd.api.types.is_datetime64_any_dtype(df[col])]
    
    print("\n=== Numeric Columns ===")
    for idx in numeric_columns:
        print(f"{idx}: {df.columns[idx]}")
    
    while True:
        print("\n=== Query Options ===")
        print("1. Range Query")
        print("2. Exact Match Query")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '3':
            print("Exiting flexible query demo.")
            break
            
        if choice not in ['1', '2']:
            print("Invalid choice. Please try again.")
            continue
            
        # Get column selection
        try:
            col_input = input("\nEnter column indices to use (comma-separated): ")
            column_indices = [int(idx.strip()) for idx in col_input.split(',')]
            
            # Validate column indices
            for idx in column_indices:
                if idx < 0 or idx >= len(df.columns):
                    print(f"Invalid column index: {idx}")
                    continue
                    
            if choice == '1':  # Range Query
                # Get custom bounds for each column
                bounds = []
                for idx in column_indices:
                    col = df.columns[idx]
                    print(f"\nColumn: {col}")
                    print(f"Min value: {df[col].min()}, Max value: {df[col].max()}")
                    
                    min_val_input = input(f"Enter minimum value for {col} (or press Enter for min): ").strip()
                    max_val_input = input(f"Enter maximum value for {col} (or press Enter for max): ").strip()
                    
                    # Handle datetime columns
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        min_val = datetime_to_numeric(pd.to_datetime(min_val_input)) if min_val_input else datetime_to_numeric(df[col].min())
                        max_val = datetime_to_numeric(pd.to_datetime(max_val_input)) if max_val_input else datetime_to_numeric(df[col].max())
                    else:
                        # For numeric columns
                        try:
                            min_val = float(min_val_input) if min_val_input else df[col].min()
                            max_val = float(max_val_input) if max_val_input else df[col].max()
                        except ValueError:
                            print(f"Invalid input for {col}. Using min and max values.")
                            min_val = df[col].min()
                            max_val = df[col].max()
                    
                    bounds.append((min_val, max_val))
                
                # Build KD-Tree for specified columns
                kd_tree = build_kd_tree_for_columns(df, column_indices)
                
                print(f"\nExecuting range query with bounds:")
                for i, idx in enumerate(column_indices):
                    col = df.columns[idx]
                    print(f"- {col}: {bounds[i]}")
                    
                # Execute KD-Tree query
                start_time = time.time()
                kd_results = kd_tree.get_points_within_bounds(bounds)
                kd_query_time = time.time() - start_time
                print(f"\nKD-Tree query completed in {kd_query_time:.4f} seconds.")
                print(f"Found {len(kd_results)} matching records")
                
                # Execute brute force query
                start_time = time.time()
                brute_force_results = kd_tree.brute_force_range_search(bounds)
                brute_force_time = time.time() - start_time
                print(f"Brute force search completed in {brute_force_time:.4f} seconds.")
                print(f"Found {len(brute_force_results)} matching records")
                
                # Performance comparison
                if brute_force_time > 0 and kd_query_time > 0:
                    print(f"Speedup factor: {brute_force_time / kd_query_time:.2f}x")
                
                # Show sample results
                print("\nSample results:")
                for row in kd_results[:5]:
                    print(row)
                if len(kd_results) > 5:
                    print(f"... and {len(kd_results) - 5} more")
                
                # Plot performance comparison
                labels = ['KD-Tree', 'Brute Force']
                times = [kd_query_time, brute_force_time]
                
                plt.figure(figsize=(8, 4))
                plt.bar(labels, times, color=['blue', 'orange'])
                plt.ylabel('Time (seconds)')
                plt.title('Query Performance Comparison')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                for i, v in enumerate(times):
                    plt.text(i, v + 0.01, f"{v:.4f}s", ha='center')
                plt.tight_layout()
                plt.show()
                
            elif choice == '2':  # Exact Match Query
                # Get match values for each column
                match_values = []
                for idx in column_indices:
                    col = df.columns[idx]
                    print(f"\nColumn: {col}")
                    
                    # Print a few sample values
                    print(f"Sample values: {df[col].sample(n=5).tolist()}")
                    
                    value_input = input(f"Enter exact value for {col} (or 'any' to match any): ").strip()
                    
                    if value_input.lower() == 'any':
                        match_values.append(None)
                    else:
                        # Handle datetime columns
                        if pd.api.types.is_datetime64_any_dtype(df[col]):
                            try:
                                value = datetime_to_numeric(pd.to_datetime(value_input))
                            except:
                                print(f"Invalid date format. Using None (match any).")
                                value = None
                        else:
                            # For numeric columns
                            try:
                                value = float(value_input) if value_input else None
                            except ValueError:
                                print(f"Invalid numeric value. Using None (match any).")
                                value = None
                        
                        match_values.append(value)
                
                # Build KD-Tree for specified columns
                kd_tree = build_kd_tree_for_columns(df, column_indices)
                
                print(f"\nExecuting exact match query with values:")
                for i, idx in enumerate(column_indices):
                    col = df.columns[idx]
                    value = match_values[i]
                    if value is None:
                        print(f"- {col}: Any value")
                    else:
                        print(f"- {col}: {value}")
                
                # Execute KD-Tree query
                start_time = time.time()
                kd_results = kd_tree.get_exact_matches(match_values)
                kd_query_time = time.time() - start_time
                print(f"\nKD-Tree query completed in {kd_query_time:.4f} seconds.")
                print(f"Found {len(kd_results)} matching records")
                
                # Execute brute force query
                start_time = time.time()
                brute_force_results = kd_tree.brute_force_exact_match(match_values)
                brute_force_time = time.time() - start_time
                print(f"Brute force search completed in {brute_force_time:.4f} seconds.")
                print(f"Found {len(brute_force_results)} matching records")
                
                # Performance comparison
                if brute_force_time > 0 and kd_query_time > 0:
                    print(f"Speedup factor: {brute_force_time / kd_query_time:.2f}x")
                
                # Show sample results
                print("\nSample results:")
                for row in kd_results[:5]:
                    print(row)
                if len(kd_results) > 5:
                    print(f"... and {len(kd_results) - 5} more")
                
                # Plot performance comparison
                labels = ['KD-Tree', 'Brute Force']
                times = [kd_query_time, brute_force_time]
                
                plt.figure(figsize=(8, 4))
                plt.bar(labels, times, color=['blue', 'orange'])
                plt.ylabel('Time (seconds)')
                plt.title('Exact Match Query Performance Comparison')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                for i, v in enumerate(times):
                    plt.text(i, v + 0.01, f"{v:.4f}s", ha='center')
                plt.tight_layout()
                plt.show()
                
        except ValueError as e:
            print(f"Input error: {e}")
            print("Please enter valid numeric values.")
        except Exception as e:
            print(f"Error during query execution: {e}")

    print("\n=== Flexible Query Demo Complete ===")
    """Main demo function for flexible queries"""
    # Map column names to indices
    columns = {name: i for i, name in enumerate(df.columns)}
    
    print("\n=== Available Columns ===")
    for name, idx in columns.items():
        print(f"{idx}: {name}")
    
    # Numeric columns (excluding dates)
    numeric_columns = [i for i, col in enumerate(df.columns) if 
                      pd.api.types.is_numeric_dtype(df[col]) and 
                      not pd.api.types.is_datetime64_any_dtype(df[col])]
    
    print("\n=== Numeric Columns ===")
    for idx in numeric_columns:
        print(f"{idx}: {df.columns[idx]}")
    
    # Demo 1: Range query on l_orderkey and l_quantity
    print("\n\n=== Demo 1: Range Query on orderkey and quantity ===")
    range_columns = [columns.get("l_orderkey", 0), columns.get("l_quantity", 3)]
    run_range_query_demo(df, range_columns)
    
    # Demo 2: Range query on all numeric columns
    print("\n\n=== Demo 2: Range Query on all numeric columns ===")
    run_range_query_demo(df, numeric_columns)
    
    # Demo 3: Exact match query on l_partkey and l_suppkey
    print("\n\n=== Demo 3: Exact Match Query on partkey and suppkey ===")
    match_columns = [columns.get("l_partkey", 1), columns.get("l_suppkey", 2)]
    run_exact_match_demo(df, match_columns)
    
    # Demo 4: Custom columns for range query
    print("\n\n=== Demo 4: Custom Columns Range Query ===")
    custom_columns = [columns.get("l_suppkey", 2), columns.get("l_extendedprice", 4)]
    run_range_query_demo(df, custom_columns)
    
    # Demo 5: Custom columns for exact match query
    print("\n\n=== Demo 5: Custom Columns Exact Match Query ===")
    custom_columns = [columns.get("l_orderkey", 0), columns.get("l_partkey", 1), columns.get("l_quantity", 3)]
    run_exact_match_demo(df, custom_columns)
    
    # Demo 6: Interactive custom query
    print("\n\n=== Demo 6: Interactive Custom Query ===")
    run_custom_query(df)
    
    print("\n=== Flexible Query Demo Complete ===")


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
    
    try:
        # First try to load all columns
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error loading all columns: {e}")
        print("Trying to load specific columns only...")
        try:
            # If that fails, try to load just the numeric columns
            df = pd.read_excel(file_path, usecols=["l_orderkey", "l_partkey", "l_suppkey", 
                                                  "l_quantity", "l_extendedprice"])
        except Exception as e2:
            print(f"Error loading specific columns: {e2}")
            print("Unable to load data. Please check the file path and format.")
            sys.exit(1)
    
    print(f"Loaded {len(df)} records")

    # Print sample data and dataset statistics
    print("\nSample data points:")
    for i, row in df.head(5).iterrows():
        print(row.tolist())
    
    print("\nData summary statistics:")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            median = df[col].median()
            p25 = df[col].quantile(0.25)
            p75 = df[col].quantile(0.75)
            print(f"{col}: min={min_val}, 25th={p25}, median={median}, 75th={p75}, max={max_val}")
    
    # Run the flexible query demo
    run_flexible_query_demo(df)
