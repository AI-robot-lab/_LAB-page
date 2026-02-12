# Sensor Fusion - Fuzja Sensorów

## Wprowadzenie

**Sensor Fusion** łączy dane z wielu czujników dla lepszej estymacji stanu robota. W robotyce humanoidalnej integruje IMU, kamery, LiDAR i czujniki proprioceptywne.

## Kalman Filter

### Extended Kalman Filter (EKF)

```python
import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # State
        self.x = np.zeros(state_dim)
        
        # Covariance
        self.P = np.eye(state_dim)
        
        # Process noise
        self.Q = np.eye(state_dim) * 0.01
        
        # Measurement noise
        self.R = np.eye(measurement_dim) * 0.1
    
    def predict(self, f, F, dt):
        """
        Prediction step
        
        Args:
            f: motion model function
            F: Jacobian of motion model
            dt: time step
        """
        # Predict state
        self.x = f(self.x, dt)
        
        # Predict covariance
        F_jac = F(self.x, dt)
        self.P = F_jac @ self.P @ F_jac.T + self.Q
    
    def update(self, z, h, H):
        """
        Update step
        
        Args:
            z: measurement
            h: measurement model function
            H: Jacobian of measurement model
        """
        # Innovation
        y = z - h(self.x)
        
        # Innovation covariance
        H_jac = H(self.x)
        S = H_jac @ self.P @ H_jac.T + self.R
        
        # Kalman gain
        K = self.P @ H_jac.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H_jac) @ self.P

# Example: IMU + GPS fusion
class IMU_GPS_Fusion:
    def __init__(self):
        # State: [x, y, vx, vy, heading]
        self.ekf = ExtendedKalmanFilter(state_dim=5, measurement_dim=3)
    
    def motion_model(self, x, dt):
        """
        f(x) - predict next state
        """
        x_new = x.copy()
        x_new[0] += x[2] * dt  # x += vx * dt
        x_new[1] += x[3] * dt  # y += vy * dt
        return x_new
    
    def motion_jacobian(self, x, dt):
        """
        F - Jacobian of motion model
        """
        F = np.eye(5)
        F[0, 2] = dt
        F[1, 3] = dt
        return F
    
    def measurement_model(self, x):
        """
        h(x) - predict measurement from state
        """
        return np.array([x[0], x[1], x[4]])  # GPS: [x, y, heading]
    
    def measurement_jacobian(self, x):
        """
        H - Jacobian of measurement model
        """
        H = np.zeros((3, 5))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 4] = 1  # heading
        return H
    
    def update(self, imu_data, gps_data, dt):
        """
        Fuse IMU and GPS
        """
        # Predict with IMU
        self.ekf.predict(self.motion_model, self.motion_jacobian, dt)
        
        # Update with GPS
        if gps_data is not None:
            z = np.array([gps_data['x'], gps_data['y'], gps_data['heading']])
            self.ekf.update(z, self.measurement_model, self.measurement_jacobian)
        
        return self.ekf.x
```

## Unscented Kalman Filter (UKF)

```python
class UnscentedKalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.n = state_dim
        self.m = measurement_dim
        
        # State and covariance
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)
        
        # Noise covariances
        self.Q = np.eye(state_dim) * 0.01
        self.R = np.eye(measurement_dim) * 0.1
        
        # UKF parameters
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0
        
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
    
    def generate_sigma_points(self):
        """
        Generate sigma points
        """
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        
        # Mean
        sigma_points[0] = self.x
        
        # Sqrt of covariance
        U = np.linalg.cholesky((self.n + self.lambda_) * self.P)
        
        for i in range(self.n):
            sigma_points[i + 1] = self.x + U[:, i]
            sigma_points[i + 1 + self.n] = self.x - U[:, i]
        
        return sigma_points
    
    def compute_weights(self):
        """
        Compute weights for sigma points
        """
        W_m = np.zeros(2 * self.n + 1)
        W_c = np.zeros(2 * self.n + 1)
        
        W_m[0] = self.lambda_ / (self.n + self.lambda_)
        W_c[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2 * self.n + 1):
            W_m[i] = 1 / (2 * (self.n + self.lambda_))
            W_c[i] = 1 / (2 * (self.n + self.lambda_))
        
        return W_m, W_c
    
    def predict(self, f, dt):
        """
        Prediction step using unscented transform
        """
        # Generate sigma points
        sigma_points = self.generate_sigma_points()
        
        # Propagate through motion model
        sigma_points_pred = np.array([f(sp, dt) for sp in sigma_points])
        
        # Weights
        W_m, W_c = self.compute_weights()
        
        # Predicted mean
        self.x = np.sum(W_m[:, None] * sigma_points_pred, axis=0)
        
        # Predicted covariance
        self.P = self.Q.copy()
        for i in range(2 * self.n + 1):
            diff = sigma_points_pred[i] - self.x
            self.P += W_c[i] * np.outer(diff, diff)
```

## Multi-Sensor Fusion

### Camera + LiDAR Fusion

```python
class CameraLidarFusion:
    def __init__(self, camera_matrix, lidar_to_camera_transform):
        self.K = camera_matrix
        self.T = lidar_to_camera_transform
    
    def project_lidar_to_image(self, lidar_points):
        """
        Project 3D LiDAR points to 2D image
        """
        # Transform to camera frame
        points_cam = self.T @ np.vstack([lidar_points.T, np.ones(len(lidar_points))])
        points_cam = points_cam[:3, :]
        
        # Project to image
        points_2d = self.K @ points_cam
        points_2d = points_2d[:2, :] / points_2d[2, :]
        
        return points_2d.T
    
    def fuse_detections(self, camera_detections, lidar_points):
        """
        Fuse 2D detections with 3D LiDAR points
        """
        fused = []
        
        # Project LiDAR to image
        points_2d = self.project_lidar_to_image(lidar_points)
        
        for det in camera_detections:
            bbox = det['bbox']
            
            # Find LiDAR points within bbox
            mask = (
                (points_2d[:, 0] >= bbox[0]) &
                (points_2d[:, 0] <= bbox[2]) &
                (points_2d[:, 1] >= bbox[1]) &
                (points_2d[:, 1] <= bbox[3])
            )
            
            points_in_bbox = lidar_points[mask]
            
            if len(points_in_bbox) > 0:
                # Estimate 3D position (median of points)
                position_3d = np.median(points_in_bbox, axis=0)
                
                fused.append({
                    'class': det['class'],
                    'bbox_2d': bbox,
                    'position_3d': position_3d,
                    'confidence': det['confidence']
                })
        
        return fused
```

## Complementary Filtering

```python
class ComplementaryFilterMultiSensor:
    def __init__(self):
        self.alpha_gyro = 0.98
        self.alpha_gps = 0.8
        
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
    
    def update(self, imu, gps, odom, dt):
        """
        Fuse IMU, GPS, and odometry
        """
        # High-frequency: IMU integration
        accel = imu['acceleration'][:2]
        self.velocity += accel * dt
        pos_imu = self.position + self.velocity * dt
        
        # Low-frequency: GPS
        if gps is not None:
            pos_gps = np.array([gps['x'], gps['y']])
            
            # Complementary filter
            self.position = self.alpha_gps * pos_gps + (1 - self.alpha_gps) * pos_imu
        else:
            self.position = pos_imu
        
        # Medium-frequency: odometry
        if odom is not None:
            vel_odom = np.array([odom['vx'], odom['vy']])
            self.velocity = 0.5 * self.velocity + 0.5 * vel_odom
        
        return self.position, self.velocity
```

## Powiązane Artykuły

- [IMU](#wiki-imu)
- [SLAM](#wiki-slam)
- [LiDAR 3D](#wiki-lidar)

---

*Ostatnia aktualizacja: 2025-02-11*
