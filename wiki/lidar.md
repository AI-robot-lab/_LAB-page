# LiDAR 3D - Light Detection and Ranging

## Wprowadzenie

**LiDAR** (Light Detection and Ranging) to technologia wykorzystująca impulsy laserowe do pomiaru odległości i tworzenia trójwymiarowych map otoczenia. W robotyce humanoidalnej LiDAR 3D jest kluczowym sensorem do nawigacji, mapowania i wykrywania przeszkód.

## Zasada Działania

### Time-of-Flight (ToF)

```
Laser → Obiekt → Odbicie → Detektor

Odległość = (Czas * Prędkość światła) / 2
```

LiDAR wysyła impulsy świetlne i mierzy czas powrotu odbicia:

```python
import numpy as np

def calculate_distance(time_of_flight):
    """
    Obliczanie odległości na podstawie ToF
    
    Args:
        time_of_flight: Czas w nanosekundach
    
    Returns:
        Odległość w metrach
    """
    SPEED_OF_LIGHT = 299_792_458  # m/s
    
    # Konwersja ns -> s
    time_seconds = time_of_flight * 1e-9
    
    # Odległość (tam i z powrotem, więc /2)
    distance = (time_seconds * SPEED_OF_LIGHT) / 2
    
    return distance

# Przykład
tof_ns = 6.67  # ~1 metr
distance = calculate_distance(tof_ns)
print(f"Odległość: {distance:.2f} m")
```

## Livox MID360 - Specyfikacja

Robot Unitree G1 wyposażony jest w **Livox MID360**:

| Parameter | Wartość |
|-----------|---------|
| **FOV** | 360° × 59° |
| **Zasięg** | 0.05 - 70 m |
| **Punkty/s** | 200,000 |
| **Dokładność** | ±2 cm @ 20m |
| **Powtarzalność** | ±1 cm |
| **Długość fali** | 905 nm |
| **Klasa lasera** | 1 (bezpieczny dla oczu) |
| **Zasilanie** | 10W |
| **Waga** | 265g |
| **IP Rating** | IP65 (pyłoszczelny, wodoodporny) |

### Non-Repetitive Scanning

Livox wykorzystuje unikalne skanowanie nie-powtarzalne:

```
Tradycyjny LiDAR:  ▓▓▓▓▓▓▓▓  (uniform grid)
Livox MID360:      ▓░▓░░▓▓░  (random pattern)
```

Zalety:
- Lepsze pokrycie w czasie
- Unikanie "ślepych punktów"
- Efektywne wykorzystanie punktów

## Przetwarzanie Danych LiDAR

### 1. Czytanie Point Cloud

```python
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

class LidarProcessor:
    def __init__(self):
        self.points = None
    
    def pointcloud_callback(self, msg: PointCloud2):
        """
        Konwersja PointCloud2 do numpy array
        """
        # Ekstract points
        points_list = []
        
        for point in pc2.read_points(
            msg, 
            field_names=("x", "y", "z", "intensity"), 
            skip_nans=True
        ):
            points_list.append([point[0], point[1], point[2], point[3]])
        
        self.points = np.array(points_list)
        
        print(f"Received {len(self.points)} points")
```

### 2. Filtrowanie

```python
def filter_pointcloud(points, min_range=0.1, max_range=50.0):
    """
    Filtrowanie punktów po odległości
    """
    # Oblicz odległość od sensora
    distances = np.linalg.norm(points[:, :3], axis=1)
    
    # Filtruj
    mask = (distances >= min_range) & (distances <= max_range)
    filtered = points[mask]
    
    return filtered

def remove_ground_plane(points, height_threshold=0.1):
    """
    Usunięcie płaszczyzny podłogi
    """
    # Załóżmy że Z to wysokość
    mask = points[:, 2] > height_threshold
    return points[mask]

def voxel_downsample(points, voxel_size=0.05):
    """
    Downsampling z użyciem voxel grid
    """
    # Dyskretyzacja do voxeli
    voxel_indices = np.floor(points[:, :3] / voxel_size).astype(int)
    
    # Unikalne voxele
    unique_voxels, inverse_indices = np.unique(
        voxel_indices, 
        axis=0, 
        return_inverse=True
    )
    
    # Średnia punktów w każdym voxelu
    downsampled = []
    for i in range(len(unique_voxels)):
        voxel_points = points[inverse_indices == i]
        centroid = np.mean(voxel_points, axis=0)
        downsampled.append(centroid)
    
    return np.array(downsampled)
```

### 3. Segmentacja

```python
from sklearn.cluster import DBSCAN

def cluster_objects(points, eps=0.5, min_samples=10):
    """
    Clustering DBSCAN dla detekcji obiektów
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(points[:, :3])
    
    # Grupuj punkty według klastrów
    clusters = []
    for label in set(labels):
        if label == -1:  # Noise
            continue
        
        cluster_points = points[labels == label]
        clusters.append(cluster_points)
    
    return clusters

def extract_bounding_boxes(clusters):
    """
    Ekstract bounding boxes z klastrów
    """
    boxes = []
    
    for cluster in clusters:
        min_point = np.min(cluster[:, :3], axis=0)
        max_point = np.max(cluster[:, :3], axis=0)
        
        center = (min_point + max_point) / 2
        size = max_point - min_point
        
        boxes.append({
            'center': center,
            'size': size,
            'num_points': len(cluster)
        })
    
    return boxes
```

## SLAM z LiDAR

### Odometry Estimation

```python
import open3d as o3d

class LidarOdometry:
    def __init__(self):
        self.previous_cloud = None
        self.cumulative_transform = np.eye(4)
    
    def estimate_transform(self, current_cloud):
        """
        ICP dla estymacji ruchu
        """
        if self.previous_cloud is None:
            self.previous_cloud = current_cloud
            return np.eye(4)
        
        # ICP registration
        reg_result = o3d.pipelines.registration.registration_icp(
            current_cloud,
            self.previous_cloud,
            max_correspondence_distance=0.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        transform = reg_result.transformation
        
        # Aktualizacja
        self.cumulative_transform = self.cumulative_transform @ transform
        self.previous_cloud = current_cloud
        
        return transform
```

### LOAM (Lidar Odometry and Mapping)

```python
def extract_edge_features(points, curvature_threshold=0.1):
    """
    Ekstract edge features (krawędzie obiektów)
    """
    edges = []
    
    # Oblicz krzywizn dla każdego punktu
    for i in range(5, len(points) - 5):
        neighbors = points[i-5:i+5]
        
        # Aproksymacja krzywizny
        diff = neighbors - points[i]
        curvature = np.linalg.norm(np.sum(diff, axis=0))
        
        if curvature > curvature_threshold:
            edges.append(points[i])
    
    return np.array(edges)

def extract_planar_features(points, planarity_threshold=0.01):
    """
    Ekstract planar features (płaskie powierzchnie)
    """
    planes = []
    
    # PCA dla lokalnych obszarów
    for i in range(10, len(points) - 10):
        neighbors = points[i-10:i+10]
        
        # Covariance matrix
        cov = np.cov(neighbors.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        
        # Planarity = lambda_min / (lambda_1 + lambda_2 + lambda_3)
        planarity = eigenvalues[0] / np.sum(eigenvalues)
        
        if planarity < planarity_threshold:
            planes.append(points[i])
    
    return np.array(planes)
```

## Wizualizacja

### Open3D Visualization

```python
import open3d as o3d

def visualize_pointcloud(points, colors=None):
    """
    Interaktywna wizualizacja point cloud
    """
    # Utwórz point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Koloruj według wysokości
        z_values = points[:, 2]
        colors = plt.cm.viridis((z_values - z_values.min()) / (z_values.max() - z_values.min()))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Wizualizacja
    o3d.visualization.draw_geometries([pcd])

def visualize_with_bboxes(points, bboxes):
    """
    Wizualizacja z bounding boxes
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    geometries = [pcd]
    
    # Dodaj bounding boxes
    for bbox in bboxes:
        # Utwórz oriented bounding box
        obb = o3d.geometry.OrientedBoundingBox(
            center=bbox['center'],
            R=np.eye(3),
            extent=bbox['size']
        )
        obb.color = (1, 0, 0)  # Red
        geometries.append(obb)
    
    o3d.visualization.draw_geometries(geometries)
```

### RViz2 Visualization (ROS2)

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray, Marker
import sensor_msgs_py.point_cloud2 as pc2

class LidarVisualizer(Node):
    def __init__(self):
        super().__init__('lidar_visualizer')
        
        # Publisher
        self.cloud_pub = self.create_publisher(
            PointCloud2, '/visualization/pointcloud', 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, '/visualization/objects', 10
        )
    
    def publish_pointcloud(self, points):
        """
        Publikuj point cloud do RViz
        """
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'lidar'
        
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        cloud_msg = pc2.create_cloud(header, fields, points)
        self.cloud_pub.publish(cloud_msg)
    
    def publish_bboxes(self, bboxes):
        """
        Publikuj bounding boxes jako markery
        """
        marker_array = MarkerArray()
        
        for i, bbox in enumerate(bboxes):
            marker = Marker()
            marker.header.frame_id = 'lidar'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'objects'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position.x = bbox['center'][0]
            marker.pose.position.y = bbox['center'][1]
            marker.pose.position.z = bbox['center'][2]
            
            marker.scale.x = bbox['size'][0]
            marker.scale.y = bbox['size'][1]
            marker.scale.z = bbox['size'][2]
            
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)
```

## Obstacle Detection

```python
class ObstacleDetector:
    def __init__(self, robot_radius=0.3):
        self.robot_radius = robot_radius
    
    def detect_obstacles(self, points):
        """
        Detekcja przeszkód w przestrzeni robota
        """
        # Grid-based representation
        grid_resolution = 0.1  # 10cm
        
        # Utwórz occupancy grid
        x_min, x_max = -10, 10
        y_min, y_max = -10, 10
        
        grid_x = int((x_max - x_min) / grid_resolution)
        grid_y = int((y_max - y_min) / grid_resolution)
        
        occupancy_grid = np.zeros((grid_x, grid_y))
        
        # Wypełnij grid
        for point in points:
            x, y = point[0], point[1]
            
            if x_min <= x < x_max and y_min <= y < y_max:
                grid_i = int((x - x_min) / grid_resolution)
                grid_j = int((y - y_min) / grid_resolution)
                
                occupancy_grid[grid_i, grid_j] = 1
        
        return occupancy_grid
    
    def find_free_space(self, occupancy_grid):
        """
        Znajdź wolną przestrzeń do nawigacji
        """
        # Erosion dla robot radius
        from scipy.ndimage import binary_erosion
        
        structuring_element = np.ones((
            int(self.robot_radius * 2 / 0.1),
            int(self.robot_radius * 2 / 0.1)
        ))
        
        free_space = binary_erosion(
            1 - occupancy_grid,
            structure=structuring_element
        )
        
        return free_space
```

## Multi-LiDAR Fusion

```python
def fuse_multiple_lidars(lidar_data_list, transforms):
    """
    Fuzja danych z wielu LiDARów
    
    Args:
        lidar_data_list: Lista point clouds
        transforms: Lista transformacji [x, y, z, roll, pitch, yaw]
    """
    fused_cloud = []
    
    for points, transform in zip(lidar_data_list, transforms):
        # Transform matrix
        T = create_transform_matrix(*transform)
        
        # Transform points
        ones = np.ones((len(points), 1))
        homogeneous = np.hstack([points[:, :3], ones])
        transformed = (T @ homogeneous.T).T[:, :3]
        
        fused_cloud.append(transformed)
    
    # Połącz wszystkie
    return np.vstack(fused_cloud)

def create_transform_matrix(x, y, z, roll, pitch, yaw):
    """
    Utwórz macierz transformacji 4x4
    """
    # Rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R = R_z @ R_y @ R_x
    
    # 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    return T
```

## Porównanie LiDARów

| Model | Punkty/s | FOV | Zasięg | Cena | Zastosowanie |
|-------|----------|-----|--------|------|--------------|
| **Livox MID360** | 200K | 360°×59° | 70m | $$ | Robotyka mobilna |
| **Velodyne VLP-16** | 300K | 360°×30° | 100m | $$$ | Autonomiczne pojazdy |
| **Ouster OS1-64** | 1.3M | 360°×45° | 120m | $$$$ | High-end robotyka |
| **RoboSense RS-LiDAR-16** | 320K | 360°×30° | 150m | $$ | Automotive |

## Powiązane Artykuły

- [Computer Vision](#wiki-computer-vision) - fuzja LiDAR + kamera
- [SLAM](#wiki-slam) - mapowanie i lokalizacja
- [ROS2](#wiki-ros2) - integracja sensora

## Zasoby

- [Livox SDK](https://github.com/Livox-SDK/Livox-SDK2)
- [Point Cloud Library (PCL)](https://pointclouds.org/)
- [Open3D](http://www.open3d.org/)

---

*Ostatnia aktualizacja: 2025-02-10*  
*Autor: Zespół Percepcji, Laboratorium Robotów Humanoidalnych*
