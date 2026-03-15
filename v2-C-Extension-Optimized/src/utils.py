"""
Utility functions for k-means clustering.
"""

import numpy as np


def validate_data(data):
    """
    Validate input data format and dimensions.
    
    Args:
        data: Input data (list or numpy array)
        
    Returns:
        Validated numpy array
        
    Raises:
        ValueError: If data is invalid
    """
    data = np.asarray(data, dtype=np.float64)
    
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional (N samples × D features)")
    
    if data.shape[0] == 0:
        raise ValueError("Data cannot be empty")
    
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")
    
    return data


def validate_parameters(k, max_iter, epsilon, n_samples):
    """
    Validate clustering parameters.
    
    Args:
        k: Number of clusters
        max_iter: Maximum iterations
        epsilon: Convergence threshold
        n_samples: Number of samples
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(k, int) or k <= 1:
        raise ValueError(f"K must be integer > 1, got {k}")
    
    if k >= n_samples:
        raise ValueError(f"K ({k}) must be < N ({n_samples})")
    
    if not isinstance(max_iter, int) or max_iter <= 1 or max_iter >= 1000:
        raise ValueError(f"max_iter must be in [2, 999], got {max_iter}")
    
    if not isinstance(epsilon, float) or epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")


def compute_inertia(data, centroids, labels):
    """
    Compute within-cluster sum of squares (inertia).
    
    Args:
        data: Data points (N × D)
        centroids: Cluster centroids (K × D)
        labels: Cluster assignments (N,)
        
    Returns:
        Inertia value
    """
    inertia = 0.0
    for k in range(len(centroids)):
        cluster_points = data[labels == k]
        if len(cluster_points) > 0:
            distances = np.linalg.norm(
                cluster_points - centroids[k],
                axis=1
            )
            inertia += np.sum(distances ** 2)
    return inertia


def normalize_data(data):
    """
    Normalize data to zero mean and unit variance.
    
    Args:
        data: Input data (N × D)
        
    Returns:
        Normalized data, mean, std
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    normalized = (data - mean) / std
    return normalized, mean, std


def denormalize_centroids(centroids, mean, std):
    """
    Convert normalized centroids back to original scale.
    
    Args:
        centroids: Normalized centroids
        mean: Original data mean
        std: Original data std
        
    Returns:
        Denormalized centroids
    """
    return centroids * std + mean
