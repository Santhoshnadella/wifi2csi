import plotly.graph_objs as go
import numpy as np
import cv2

def create_point_cloud_figure(points, title="WiFi-Sensed Point Cloud"):
    """
    Creates a Plotly 3D scatter plot.
    points: [N, 3] array
    """
    if points is None or len(points) == 0:
        # Return empty placeholder
        return go.Figure()

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=z,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-3, 3]),
            yaxis=dict(range=[-3, 3]),
            zaxis=dict(range=[-1, 3]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig = go.Figure(data=[trace], layout=layout)
    return fig

def overlay_points_on_image(image, points, camera_stats=None):
    """
    Projects 3D points onto the 2D image plane (Webcam).
    Simple perspective projection.
    
    image: OpenCV image (BGR)
    points: [N, 3]
    """
    h, w, _ = image.shape
    
    # Simple pinhole camera intrinsics approximation
    f = w  # Focal length
    cx, cy = w / 2, h / 2
    
    viz_img = image.copy()
    
    if points is None or len(points) == 0:
        return viz_img
    
    # Filter points behind camera
    valid_points = points[points[:, 1] > 0.1] # Assume Y is forward
    
    if len(valid_points) == 0:
        return viz_img

    # Project
    # X_img = f * (X / Y) + cx
    # Y_img = f * (Z / Y) + cy (Assuming Z is up in world, but Y is up in image? Adjusting coords)
    
    # Standard: X right, Y down, Z forward for camera
    # Our World: X right, Y forward, Z up
    
    # Map World (x, y, z) -> Camera (x_c, y_c, z_c)
    # x_c = x
    # y_c = -z (Z up in world maps to negative Y in camera to be 'up')
    # z_c = y (Y forward in world maps to Z depth in camera)
    
    x_c = valid_points[:, 0]
    y_c = -valid_points[:, 2] + 1.0 # Shift down a bit
    z_c = valid_points[:, 1]
    
    u = (f * (x_c / z_c) + cx).astype(int)
    v = (f * (y_c / z_c) + cy).astype(int)
    
    # Draw
    for i in range(len(u)):
        if 0 <= u[i] < w and 0 <= v[i] < h:
            # Distance based color / size
            dist = z_c[i]
            radius = max(2, int(10 / dist))
            cv2.circle(viz_img, (u[i], v[i]), radius, (0, 255, 0), -1)
            
    # Add Text
    cv2.putText(viz_img, "Through-Wall Vision Active", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    return viz_img
