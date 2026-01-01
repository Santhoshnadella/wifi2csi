import numpy as np
from sklearn.cluster import DBSCAN
import time
import math

def generate_synthetic_csi(batch_size=1, num_antennas=3, num_subcarriers=114, time_steps=10):
    """
    Generates synthetic CSI data for testing/demo purposes.
    Simulates a 'breathing' or moving variance to make it look alive.
    
    Returns:
        np.array: Shape (batch_size, num_antennas, num_subcarriers, 2, time_steps)
                  The '2' represents (Real, Imaginary) parts.
    """
    # Create base complex Gaussian noise
    real_part = np.random.normal(0, 1, (batch_size, num_antennas, num_subcarriers, time_steps))
    imag_part = np.random.normal(0, 1, (batch_size, num_antennas, num_subcarriers, time_steps))
    
    # Add a time-varying component to simulate movement
    t = time.time()
    for i in range(num_antennas):
        modulation = np.sin(t * 2 + i) * 0.5
        real_part[:, i, :, :] += modulation
        imag_part[:, i, :, :] += np.cos(t * 2 + i) * 0.5

    # Stack to match model input expectation: [B, Ant, Sub, 2, Time]
    # Note: Model expects (Real, Imag) at dim 3
    csi_stack = np.stack([real_part, imag_part], axis=3)
    return csi_stack.astype(np.float32)

def generate_fake_human_points(num_points=50, center=(0, 0, 1.7), scale=0.5):
    """
    Generates a cluster of points resembling a human shape for DEMO visualization
    when no real model weight is available.
    """
    # Torso (cylinder-ish)
    points = []
    t_now = time.time()
    
    # Simulate "walking" bobbing
    bob = np.sin(t_now * 5) * 0.05
    center = (center[0] + np.sin(t_now)*0.5, center[1], center[2] + bob)
    
    for i in range(num_points):
        # Random Gaussian blob
        x = np.random.normal(center[0], 0.2 * scale)
        y = np.random.normal(center[1], 0.2 * scale)
        z = np.random.normal(center[2], 0.6 * scale)
        points.append([x, y, z])
    
    return np.array(points)

def analyze_point_cloud(points):
    """
    Performs DBSCAN clustering to find 'people' and describes the environment.
    points: [N, 3] numpy array
    """
    if len(points) == 0:
        return {"count": 0, "description": "Empty room", "clusters": []}

    # Clustering
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(points)
    labels = clustering.labels_
    
    # Number of unique clusters (ignoring noise -1)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    num_people = len(unique_labels)
    
    # Simple descriptor
    desc = "Static Environment"
    if num_people == 1:
        desc = "Single person detected"
    elif num_people > 1:
        desc = f"{num_people} people detected"
    
    return {
        "count": num_people,
        "description": desc,
        "labels": labels
    }

def estimate_room_skeleton(points):
    """
    Estimates room bounds from points.
    """
    if len(points) == 0:
        return None
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    return min_bound, max_bound
