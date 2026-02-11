# SLAM - Simultaneous Localization and Mapping

## Wprowadzenie

**SLAM (Simultaneous Localization and Mapping)** to problem jednoczesnej lokalizacji robota i budowy mapy otoczenia. Jest fundamentalny dla autonomicznej nawigacji robotów humanoidalnych w nieznanych środowiskach.

## Problem SLAM

### Matematyczna Formulacja

```python
# Stan robota
x_t = [x, y, θ]  # pozycja i orientacja

# Mapa
m = {landmark_1, landmark_2, ..., landmark_n}

# Cel: Estymuj p(x_t, m | z_{1:t}, u_{1:t})
# gdzie:
# z_{1:t} - pomiary sensorów
# u_{1:t} - komendy sterowania
```

### Extended Kalman Filter SLAM (EKF-SLAM)

```python
import numpy as np

class EKF_SLAM:
    def __init__(self):
        # Stan: [robot_x, robot_y, robot_theta, landmark_1_x, landmark_1_y, ...]
        self.state = np.zeros(3)  # Początkowo tylko robot
        
        # Macierz kowariancji
        self.P = np.eye(3) * 0.1
        
        # Noise models
        self.Q = np.diag([0.1, 0.1, 0.05])  # Process noise
        self.R = np.diag([0.5, 0.1])  # Measurement noise (range, bearing)
        
        self.landmarks = {}  # ID -> index w state
    
    def predict(self, v, w, dt):
        """
        Predykcja ruchu robota
        v - prędkość liniowa
        w - prędkość kątowa
        dt - czas
        """
        x, y, theta = self.state[:3]
        
        # Motion model
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + w * dt
        
        # Update state
        self.state[:3] = [x_new, y_new, theta_new]
        
        # Jacobian of motion model
        G = np.eye(len(self.state))
        G[0, 2] = -v * np.sin(theta) * dt
        G[1, 2] = v * np.cos(theta) * dt
        
        # Update covariance
        self.P = G @ self.P @ G.T
        self.P[:3, :3] += self.Q
    
    def update(self, landmark_id, z_range, z_bearing):
        """
        Update z pomiarem landmark
        """
        if landmark_id not in self.landmarks:
            # Initialize new landmark
            self.add_landmark(landmark_id, z_range, z_bearing)
            return
        
        # Get landmark position from state
        lm_idx = self.landmarks[landmark_id]
        lm_x = self.state[lm_idx]
        lm_y = self.state[lm_idx + 1]
        
        # Robot position
        x, y, theta = self.state[:3]
        
        # Expected measurement
        dx = lm_x - x
        dy = lm_y - y
        q = dx**2 + dy**2
        
        z_hat = np.array([
            np.sqrt(q),  # expected range
            np.arctan2(dy, dx) - theta  # expected bearing
        ])
        
        # Innovation
        z = np.array([z_range, z_bearing])
        y = z - z_hat
        y[1] = self.normalize_angle(y[1])
        
        # Jacobian
        H = self.compute_jacobian(x, y, theta, lm_x, lm_y, q)
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state += K @ y
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P
    
    def add_landmark(self, landmark_id, z_range, z_bearing):
        """
        Dodaj nowy landmark do mapy
        """
        x, y, theta = self.state[:3]
        
        # Compute landmark position
        lm_x = x + z_range * np.cos(z_bearing + theta)
        lm_y = y + z_range * np.sin(z_bearing + theta)
        
        # Add to state
        new_state = np.append(self.state, [lm_x, lm_y])
        
        # Expand covariance matrix
        n = len(self.state)
        new_P = np.zeros((n + 2, n + 2))
        new_P[:n, :n] = self.P
        new_P[n:, n:] = np.eye(2) * 10.0  # High initial uncertainty
        
        self.state = new_state
        self.P = new_P
        self.landmarks[landmark_id] = n
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def compute_jacobian(self, x, y, theta, lm_x, lm_y, q):
        """Compute measurement Jacobian"""
        sqrt_q = np.sqrt(q)
        dx = lm_x - x
        dy = lm_y - y
        
        H = np.zeros((2, len(self.state)))
        
        # Robot部分
        H[0, 0] = -dx / sqrt_q
        H[0, 1] = -dy / sqrt_q
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1
        
        # Landmark部分
        lm_idx = self.landmarks[list(self.landmarks.keys())[0]]  # simplified
        H[0, lm_idx] = dx / sqrt_q
        H[0, lm_idx + 1] = dy / sqrt_q
        H[1, lm_idx] = -dy / q
        H[1, lm_idx + 1] = dx / q
        
        return H
```

## Visual SLAM

### ORB-SLAM

```python
import cv2
import numpy as np

class ORB_SLAM:
    def __init__(self):
        # ORB feature detector
        self.orb = cv2.ORB_create(nfeatures=2000)
        
        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Keyframes
        self.keyframes = []
        self.map_points = []
        
        # Camera intrinsics (example)
        self.K = np.array([
            [718.856, 0, 607.1928],
            [0, 718.856, 185.2157],
            [0, 0, 1]
        ])
    
    def process_frame(self, image):
        """
        Przetwórz nową klatkę
        """
        # Detect ORB features
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        
        if len(self.keyframes) == 0:
            # First frame - initialize
            self.initialize_map(image, keypoints, descriptors)
            return
        
        # Match with previous keyframe
        last_kf = self.keyframes[-1]
        matches = self.matcher.match(descriptors, last_kf['descriptors'])
        
        # Estimate pose
        pose = self.estimate_pose(keypoints, matches, last_kf)
        
        # Triangulate new points
        new_points = self.triangulate_points(keypoints, matches, pose)
        
        # Bundle adjustment (optional)
        # self.bundle_adjustment()
        
        # Add keyframe if needed
        if self.should_add_keyframe(matches):
            self.add_keyframe(image, keypoints, descriptors, pose)
        
        return pose
    
    def initialize_map(self, image, keypoints, descriptors):
        """
        Inicjalizuj mapę z pierwszej klatki
        """
        self.keyframes.append({
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': np.eye(4)  # Identity for first frame
        })
    
    def estimate_pose(self, keypoints, matches, last_kf):
        """
        Estymuj pozę kamery używając PnP
        """
        # Extract matched points
        pts_2d = np.float32([keypoints[m.queryIdx].pt for m in matches])
        pts_3d = np.float32([self.map_points[m.trainIdx] for m in matches])
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            pts_3d, pts_2d, self.K, None
        )
        
        # Convert to 4x4 pose matrix
        R, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec.flatten()
        
        return pose
    
    def triangulate_points(self, keypoints, matches, pose):
        """
        Trianguluj nowe punkty 3D
        """
        # Projection matrices
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ pose[:3, :]
        
        # Triangulate
        points_4d = cv2.triangulatePoints(
            P1, P2,
            pts1.T, pts2.T
        )
        
        # Convert to 3D
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T
    
    def should_add_keyframe(self, matches):
        """
        Decyzja czy dodać nową keyframe
        """
        # Add keyframe if:
        # - Too few matches with last keyframe
        # - Significant motion
        return len(matches) < 50
    
    def add_keyframe(self, image, keypoints, descriptors, pose):
        """
        Dodaj nową keyframe
        """
        self.keyframes.append({
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': pose
        })
```

## LiDAR SLAM

### LOAM (Lidar Odometry and Mapping)

```python
import numpy as np
from scipy.spatial import KDTree

class LOAM:
    def __init__(self):
        self.edge_points = []
        self.planar_points = []
        
        self.pose = np.eye(4)
        self.map_edges = []
        self.map_planes = []
    
    def extract_features(self, point_cloud):
        """
        Ekstraktuj edge i planar features z chmury punktów
        """
        # Compute curvature for each point
        curvatures = self.compute_curvature(point_cloud)
        
        # Sort by curvature
        sorted_idx = np.argsort(curvatures)
        
        # Edge features (high curvature)
        edge_idx = sorted_idx[-100:]  # Top 100
        edge_points = point_cloud[edge_idx]
        
        # Planar features (low curvature)
        planar_idx = sorted_idx[:100]  # Bottom 100
        planar_points = point_cloud[planar_idx]
        
        return edge_points, planar_points
    
    def compute_curvature(self, points):
        """
        Oblicz krzywiznę dla każdego punktu
        """
        n = len(points)
        curvatures = np.zeros(n)
        
        for i in range(5, n - 5):
            # Use neighboring points
            neighbors = points[i-5:i+6]
            
            # Center point
            center = points[i]
            
            # Sum of distances
            diff = neighbors - center
            curvature = np.linalg.norm(diff.sum(axis=0))
            
            curvatures[i] = curvature
        
        return curvatures
    
    def scan_matching(self, edge_points, planar_points):
        """
        Dopasuj scan do mapy używając ICP
        """
        # Match edge features
        edge_tree = KDTree(self.map_edges)
        edge_matches = edge_tree.query(edge_points, k=2)
        
        # Match planar features
        planar_tree = KDTree(self.map_planes)
        planar_matches = planar_tree.query(planar_points, k=3)
        
        # Compute transformation using GICP
        transform = self.compute_transform(
            edge_points, edge_matches,
            planar_points, planar_matches
        )
        
        return transform
    
    def update_map(self, edge_points, planar_points):
        """
        Aktualizuj mapę
        """
        # Transform points to global frame
        edge_global = self.transform_points(edge_points, self.pose)
        planar_global = self.transform_points(planar_points, self.pose)
        
        # Add to map
        self.map_edges.extend(edge_global)
        self.map_planes.extend(planar_global)
        
        # Downsample map (voxel grid filter)
        self.map_edges = self.voxel_downsample(self.map_edges, voxel_size=0.1)
        self.map_planes = self.voxel_downsample(self.map_planes, voxel_size=0.2)
```

## Graph SLAM

```python
import networkx as nx
from scipy.optimize import least_squares

class GraphSLAM:
    def __init__(self):
        self.graph = nx.Graph()
        self.poses = {}  # node_id -> pose
        self.landmarks = {}  # landmark_id -> position
        
        self.node_counter = 0
    
    def add_pose_node(self, pose):
        """
        Dodaj node pozycji robota
        """
        node_id = f"x{self.node_counter}"
        self.node_counter += 1
        
        self.graph.add_node(node_id, type='pose')
        self.poses[node_id] = pose
        
        return node_id
    
    def add_odometry_edge(self, from_node, to_node, relative_pose, information):
        """
        Dodaj krawędź odometrii
        """
        self.graph.add_edge(
            from_node, to_node,
            type='odometry',
            measurement=relative_pose,
            information=information
        )
    
    def add_landmark_observation(self, pose_node, landmark_id, measurement, information):
        """
        Dodaj obserwację landmark
        """
        if landmark_id not in self.landmarks:
            # Initialize landmark
            self.landmarks[landmark_id] = self.initialize_landmark(
                self.poses[pose_node], measurement
            )
            self.graph.add_node(landmark_id, type='landmark')
        
        self.graph.add_edge(
            pose_node, landmark_id,
            type='observation',
            measurement=measurement,
            information=information
        )
    
    def optimize(self):
        """
        Optymalizuj graf używając least squares
        """
        # Build state vector: [poses, landmarks]
        x0 = self.build_state_vector()
        
        # Optimize
        result = least_squares(
            self.error_function,
            x0,
            method='lm'
        )
        
        # Update poses and landmarks
        self.update_from_state(result.x)
    
    def error_function(self, state):
        """
        Funkcja błędu dla optymalizacji
        """
        errors = []
        
        for edge in self.graph.edges(data=True):
            if edge[2]['type'] == 'odometry':
                # Odometry constraint
                error = self.odometry_error(edge, state)
            else:
                # Landmark observation
                error = self.observation_error(edge, state)
            
            errors.extend(error)
        
        return np.array(errors)
```

## ROS2 Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

class SLAMNode(Node):
    def __init__(self):
        super().__init__('slam_node')
        
        # SLAM backend
        self.slam = GraphSLAM()
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # Publishers
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/slam/pose',
            10
        )
        
        self.last_pose = None
    
    def scan_callback(self, msg):
        """Process laser scan"""
        # Extract features
        features = self.extract_features_from_scan(msg)
        
        # Match and update
        self.slam.update(features)
        
        # Publish pose
        self.publish_pose()
    
    def odom_callback(self, msg):
        """Process odometry"""
        current_pose = self.odom_to_pose(msg)
        
        if self.last_pose is not None:
            # Compute relative motion
            relative = self.compute_relative_pose(self.last_pose, current_pose)
            
            # Add to graph
            self.slam.add_odometry_edge(relative)
        
        self.last_pose = current_pose
```

## Porównanie Metod SLAM

| Metoda | Typ Sensorów | Środowisko | Dokładność | Złożoność |
|--------|-------------|------------|------------|-----------|
| **EKF-SLAM** | Laser/Camera | Structured | Średnia | O(n²) |
| **FastSLAM** | Laser | Indoor | Dobra | O(M log n) |
| **ORB-SLAM** | Camera | Visual | Wysoka | O(n) |
| **LOAM** | LiDAR | Outdoor | Bardzo wysoka | O(n log n) |
| **Graph-SLAM** | Multi-sensor | Any | Najwyższa | O(n³) |

## Powiązane Artykuły

- [LiDAR 3D](#wiki-lidar)
- [Computer Vision](#wiki-computer-vision)
- [IMU - Pomiary Inercjalne](#wiki-imu)
- [Sensor Fusion](#wiki-sensor-fusion)

## Zasoby

- [SLAM Tutorial](http://ais.informatik.uni-freiburg.de/teaching/ws13/mapping/)
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [LOAM](https://github.com/laboshinl/loam_velodyne)

---

*Ostatnia aktualizacja: 2025-02-11*  
*Autor: Zespół Robotyki, Laboratorium Robotów Humanoidalnych*
