"""
Visualization utilities for clustering analysis.

Implements the Elbow Method for automatic optimal K selection.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def calculate_inertia_values(data, k_range):
    """
    Calculate inertia (within-cluster sum of squares) for different K values.
    
    Args:
        data: (N, D) array of data points
        k_range: Iterable of K values to test
        
    Returns:
        inertias: List of inertia values for each K
    """
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias


def find_elbow_point(k_values, inertias):
    """
    Find the elbow point using the perpendicular distance method.
    
    The elbow point is the K value where the curve has maximum perpendicular
    distance to the line connecting the first and last points.
    
    Args:
        k_values: Array of K values
        inertias: Array of inertia values
        
    Returns:
        optimal_k: The K value at the elbow point
        elbow_index: Index of the elbow point
    """
    # Convert to numpy arrays
    k_array = np.array(k_values)
    inertia_array = np.array(inertias)
    
    # Create 3D points for distance calculation
    n = len(k_values)
    start_point = np.array([k_array[0], inertia_array[0], 0])
    end_point = np.array([k_array[-1], inertia_array[-1], 0])
    
    # Calculate perpendicular distances
    distances = []
    for i, (k, inertia) in enumerate(zip(k_array, inertia_array)):
        point = np.array([k, inertia, 0])
        
        # Vector from start to end
        line_vec = end_point - start_point
        # Vector from start to current point
        point_vec = point - start_point
        
        # Perpendicular distance
        line_len = np.linalg.norm(line_vec)
        if line_len > 0:
            distance = np.linalg.norm(np.cross(line_vec, point_vec)) / line_len
        else:
            distance = 0
        
        distances.append(distance)
    
    # Find index of maximum distance
    elbow_index = np.argmax(distances)
    optimal_k = k_array[elbow_index]
    
    return int(optimal_k), elbow_index


def elbow_method(data, k_range=None, save_path=None):
    """
    Elbow Method for optimal K selection.
    
    Visualizes inertia vs K and automatically detects the elbow point.
    
    Args:
        data: (N, D) array of data points
        k_range: Iterable of K values (default: 1-10)
        save_path: Path to save the plot (optional)
        
    Returns:
        optimal_k: Recommended number of clusters
    """
    if k_range is None:
        k_range = range(1, 11)
    
    k_values = list(k_range)
    inertias = calculate_inertia_values(data, k_values)
    
    # Find elbow point
    optimal_k, elbow_index = find_elbow_point(k_values, inertias)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8, label='Inertia')
    plt.xlabel('K (Number of Clusters)', fontsize=12)
    plt.ylabel('Average Dispersion (Inertia)', fontsize=12)
    plt.title('Elbow Method for Selection of Optimal K Clusters', fontsize=14)
    plt.xticks(range(0, max(k_values) + 1, 1))
    plt.grid(True, alpha=0.3)
    
    # Highlight elbow point
    plt.annotate(
        f'Elbow Point (K={optimal_k})',
        xy=(k_values[elbow_index], inertias[elbow_index]),
        xytext=(k_values[elbow_index] + 1, inertias[elbow_index] + 50),
        arrowprops=dict(facecolor='red', arrowstyle='->', lw=2),
        fontsize=11,
        color='red',
        weight='bold'
    )
    
    plt.legend()
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()
    
    return optimal_k
