# Pose Estimation - Estymacja Pozy

## Wprowadzenie

**Pose Estimation** to zadanie określania pozycji i orientacji obiektów lub osób. W robotyce humanoidalnej służy do śledzenia ludzi, wykrywania gestów i kontroli własnej pozy robota.

## Human Pose Estimation

### OpenPose-Style Detection

```python
import cv2
import mediapipe as mp

class HumanPoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def estimate_pose(self, image):
        """
        Estimate 33 keypoints of human pose
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract keypoints
        landmarks = results.pose_landmarks.landmark
        
        keypoints = []
        for landmark in landmarks:
            keypoints.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        return keypoints
    
    def draw_pose(self, image, landmarks):
        """
        Draw pose skeleton on image
        """
        if landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return image

# Usage
estimator = HumanPoseEstimator()
image = cv2.imread('person.jpg')

keypoints = estimator.estimate_pose(image)
image_with_pose = estimator.draw_pose(image, keypoints)
```

## 6D Object Pose Estimation

### PnP (Perspective-n-Point)

```python
class ObjectPoseEstimator:
    def __init__(self, camera_matrix, object_points):
        """
        Args:
            camera_matrix: 3x3 intrinsic matrix
            object_points: 3D model points
        """
        self.K = camera_matrix
        self.object_points = np.array(object_points, dtype=np.float32)
        self.dist_coeffs = np.zeros(4)
    
    def estimate_pose(self, image_points):
        """
        Estimate 6D pose from 2D-3D correspondences
        
        Returns: rotation_vector, translation_vector
        """
        image_points = np.array(image_points, dtype=np.float32)
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            image_points,
            self.K,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, None
        
        return rvec, tvec
    
    def rotation_vector_to_matrix(self, rvec):
        """
        Convert rotation vector to rotation matrix
        """
        R, _ = cv2.Rodrigues(rvec)
        return R
    
    def get_pose_matrix(self, rvec, tvec):
        """
        Get 4x4 transformation matrix
        """
        R = self.rotation_vector_to_matrix(rvec)
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        
        return T

# Example: Cube detection
cube_points_3d = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
], dtype=np.float32)

camera_matrix = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float32)

estimator = ObjectPoseEstimator(camera_matrix, cube_points_3d[:4])

# Detected corners in image
image_points = np.array([
    [100, 100],
    [200, 100],
    [200, 200],
    [100, 200]
], dtype=np.float32)

rvec, tvec = estimator.estimate_pose(image_points)
pose_matrix = estimator.get_pose_matrix(rvec, tvec)
```

## Deep Learning Pose Estimation

### Custom Pose Network

```python
import torch
import torch.nn as nn

class PoseNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super().__init__()
        
        # Backbone (ResNet-like)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Heatmap head
        self.heatmap = nn.Conv2d(256, num_keypoints, kernel_size=1)
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear')
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Heatmaps
        heatmaps = self.heatmap(x)
        heatmaps = self.upsample(heatmaps)
        
        return heatmaps
    
    def predict_keypoints(self, heatmaps):
        """
        Extract keypoint coordinates from heatmaps
        """
        batch_size, num_keypoints, h, w = heatmaps.shape
        
        # Find maximum in each heatmap
        heatmaps_flat = heatmaps.view(batch_size, num_keypoints, -1)
        max_vals, max_indices = torch.max(heatmaps_flat, dim=2)
        
        # Convert to 2D coordinates
        keypoints = torch.zeros(batch_size, num_keypoints, 2)
        keypoints[:, :, 0] = max_indices % w
        keypoints[:, :, 1] = max_indices // w
        
        return keypoints, max_vals
```

## Multi-Person Pose Estimation

```python
class MultiPersonPoseEstimator:
    def __init__(self):
        self.person_detector = YOLODetector()
        self.pose_estimator = HumanPoseEstimator()
    
    def estimate_poses(self, image):
        """
        Estimate poses for multiple people
        """
        # Detect people
        detections = self.person_detector.detect(image, classes=['person'])
        
        poses = []
        
        for det in detections:
            # Crop person
            x1, y1, x2, y2 = det['bbox']
            person_img = image[y1:y2, x1:x2]
            
            # Estimate pose
            keypoints = self.pose_estimator.estimate_pose(person_img)
            
            if keypoints:
                # Transform coordinates to original image
                for kp in keypoints:
                    kp['x'] = kp['x'] * (x2 - x1) + x1
                    kp['y'] = kp['y'] * (y2 - y1) + y1
                
                poses.append({
                    'bbox': det['bbox'],
                    'keypoints': keypoints,
                    'confidence': det['confidence']
                })
        
        return poses
```

## Robot Self-Pose Estimation

```python
class RobotPoseEstimator:
    def __init__(self, robot_model):
        self.robot = robot_model
    
    def estimate_base_pose(self, joint_states, foot_contacts):
        """
        Estimate floating base pose from joint encoders and foot contacts
        """
        # Forward kinematics to feet
        left_foot_pos = self.robot.forward_kinematics('left_foot', joint_states)
        right_foot_pos = self.robot.forward_kinematics('right_foot', joint_states)
        
        # If both feet on ground, estimate base from average
        if foot_contacts['left'] and foot_contacts['right']:
            base_height = (left_foot_pos[2] + right_foot_pos[2]) / 2
        elif foot_contacts['left']:
            base_height = left_foot_pos[2]
        elif foot_contacts['right']:
            base_height = right_foot_pos[2]
        else:
            # In air - use IMU
            base_height = None
        
        return base_height
```

## Powiązane Artykuły

- [MediaPipe](#wiki-mediapipe)
- [Computer Vision](#wiki-computer-vision)
- [Object Detection](#wiki-object-detection)

---

*Ostatnia aktualizacja: 2025-02-11*
