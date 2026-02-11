# MediaPipe - Framework dla Machine Learning

## Wprowadzenie

**MediaPipe** to cross-platform framework od Google do budowania pipelines ML dla percepcji człowieka. W robotyce humanoidalnej wykorzystywany do detekcji pozy, rozpoznawania gestów i śledzenia twarzy.

## Instalacja

```bash
# Python
pip install mediapipe opencv-python

# ROS2 workspace
cd ~/ros2_ws/src
git clone https://github.com/google/mediapipe
cd mediapipe
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop:hello_world
```

## Pose Estimation - Detekcja Pozy

### Basic Usage

```python
import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def process_frame(self, image):
        """
        Wykryj pozę człowieka
        
        Returns:
            landmarks: 33 punkty ciała
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            return results.pose_landmarks
        
        return None
    
    def get_joint_angles(self, landmarks):
        """
        Oblicz kąty stawów
        """
        if not landmarks:
            return None
        
        # Get landmark positions
        shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        
        # Calculate angle
        angle = self.calculate_angle(
            [shoulder.x, shoulder.y],
            [elbow.x, elbow.y],
            [wrist.x, wrist.y]
        )
        
        return {'left_elbow': angle}
    
    def calculate_angle(self, a, b, c):
        """
        Oblicz kąt między trzema punktami
        """
        import numpy as np
        
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
```

### 33 Punkty Ciała

```python
POSE_LANDMARKS = {
    0: 'NOSE',
    1: 'LEFT_EYE_INNER',
    2: 'LEFT_EYE',
    3: 'LEFT_EYE_OUTER',
    4: 'RIGHT_EYE_INNER',
    5: 'RIGHT_EYE',
    6: 'RIGHT_EYE_OUTER',
    7: 'LEFT_EAR',
    8: 'RIGHT_EAR',
    9: 'MOUTH_LEFT',
    10: 'MOUTH_RIGHT',
    11: 'LEFT_SHOULDER',
    12: 'RIGHT_SHOULDER',
    13: 'LEFT_ELBOW',
    14: 'RIGHT_ELBOW',
    15: 'LEFT_WRIST',
    16: 'RIGHT_WRIST',
    17: 'LEFT_PINKY',
    18: 'RIGHT_PINKY',
    19: 'LEFT_INDEX',
    20: 'RIGHT_INDEX',
    21: 'LEFT_THUMB',
    22: 'RIGHT_THUMB',
    23: 'LEFT_HIP',
    24: 'RIGHT_HIP',
    25: 'LEFT_KNEE',
    26: 'RIGHT_KNEE',
    27: 'LEFT_ANKLE',
    28: 'RIGHT_ANKLE',
    29: 'LEFT_HEEL',
    30: 'RIGHT_HEEL',
    31: 'LEFT_FOOT_INDEX',
    32: 'RIGHT_FOOT_INDEX'
}
```

## Hand Tracking - Śledzenie Dłoni

```python
class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect_hands(self, image):
        """
        Wykryj dłonie
        
        Returns:
            List of hand landmarks (21 points per hand)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        hands_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Extract data
                hand_info = {
                    'landmarks': hand_landmarks,
                    'handedness': handedness.classification[0].label,
                    'score': handedness.classification[0].score
                }
                
                hands_data.append(hand_info)
        
        return hands_data
    
    def recognize_gesture(self, hand_landmarks):
        """
        Rozpoznaj gest dłoni
        """
        if not hand_landmarks:
            return "UNKNOWN"
        
        # Get finger tips and bases
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        wrist = hand_landmarks.landmark[0]
        
        # Check which fingers are extended
        fingers_up = []
        
        # Thumb (special case)
        if thumb_tip.x < hand_landmarks.landmark[3].x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
        
        # Other fingers
        for tip_id in [8, 12, 16, 20]:
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        # Recognize gestures
        total = sum(fingers_up)
        
        if total == 0:
            return "FIST"
        elif total == 5:
            return "OPEN_PALM"
        elif fingers_up == [0, 1, 0, 0, 0]:
            return "POINTING"
        elif fingers_up == [1, 1, 0, 0, 0]:
            return "PEACE"
        elif fingers_up == [0, 1, 1, 0, 0]:
            return "TWO"
        elif fingers_up == [1, 1, 1, 0, 0]:
            return "THREE"
        elif fingers_up == [1, 1, 1, 1, 0]:
            return "FOUR"
        else:
            return "OTHER"
```

## Face Mesh - Siatka Twarzy

```python
class FaceMeshDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect_face(self, image):
        """
        Wykryj siatkę twarzy (468 punktów)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw mesh
                self.mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0),
                        thickness=1,
                        circle_radius=1
                    )
                )
                
                return face_landmarks
        
        return None
    
    def get_head_pose(self, face_landmarks, image_shape):
        """
        Estymuj pozę głowy (rotation i translation)
        """
        import numpy as np
        
        h, w = image_shape[:2]
        
        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # 2D image points from landmarks
        image_points = np.array([
            (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),     # Nose
            (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h), # Chin
            (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),   # Left eye
            (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h), # Right eye
            (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),   # Left mouth
            (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h)  # Right mouth
        ], dtype="double")
        
        # Camera matrix
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        # Distortion coefficients
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # Convert rotation vector to Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rotation_mat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        return {
            'pitch': euler_angles[0][0],  # Up/down
            'yaw': euler_angles[1][0],    # Left/right
            'roll': euler_angles[2][0]    # Tilt
        }
```

## ROS2 Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

class MediaPipeNode(Node):
    def __init__(self):
        super().__init__('mediapipe_node')
        
        # Detectors
        self.pose_detector = PoseDetector()
        self.hand_detector = HandDetector()
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publishers
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/mediapipe/human_pose',
            10
        )
    
    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Detect pose
        landmarks = self.pose_detector.process_frame(cv_image)
        
        if landmarks:
            # Publish pose
            self.publish_pose(landmarks)
        
        # Detect hands
        hands = self.hand_detector.detect_hands(cv_image)
        
        # Convert back and publish annotated image
        annotated_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        # self.annotated_pub.publish(annotated_msg)
```

## Performance Tips

### GPU Acceleration

```python
# Enable GPU (if available)
pose = mp.solutions.pose.Pose(
    model_complexity=2,  # 0=Lite, 1=Full, 2=Heavy
    enable_segmentation=True,
    min_detection_confidence=0.5
)
```

### Optimization

```python
# Reduce resolution for faster processing
def resize_for_processing(image, max_size=640):
    h, w = image.shape[:2]
    
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    
    return image

# Skip frames
frame_count = 0
if frame_count % 2 == 0:  # Process every 2nd frame
    results = pose.process(image)
frame_count += 1
```

## Porównanie z Innymi Metodami

| Feature | MediaPipe | OpenPose | AlphaPose |
|---------|-----------|----------|-----------|
| **Prędkość** | ⚡⚡⚡ | ⚡⚡ | ⚡ |
| **Dokładność** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Multi-person** | ❌ | ✅ | ✅ |
| **Mobile** | ✅ | ❌ | ❌ |
| **Hand tracking** | ✅ | ❌ | ❌ |

## Powiązane Artykuły

- [Computer Vision](#wiki-computer-vision)
- [Pose Estimation](#wiki-pose-estimation)
- [OpenCV](#wiki-opencv)

## Zasoby

- [MediaPipe Docs](https://google.github.io/mediapipe/)
- [GitHub](https://github.com/google/mediapipe)

---

*Ostatnia aktualizacja: 2025-02-11*  
*Autor: Zespół Percepcji, Laboratorium Robotów Humanoidalnych*
