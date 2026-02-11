# Computer Vision - Wizja Komputerowa

## Wprowadzenie

**Computer Vision** (Wizja Komputerowa) to dziedzina sztucznej inteligencji zajmująca się umożliwieniem komputerom "widzenia" i rozumienia treści wizualnych. W robotyce humanoidalnej CV jest fundamentem dla percepcji wizualnej, rozpoznawania obiektów i interakcji ze środowiskiem.

## Podstawowe Zadania

### 1. Image Classification

Klasyfikacja całego obrazu do jednej z kategorii:

```python
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Załaduj pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Predykcja
def classify_image(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image_tensor)
    
    # Top 5 predictions
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # Załaduj labels
    with open('imagenet_classes.txt') as f:
        categories = [line.strip() for line in f.readlines()]
    
    results = []
    for i in range(top5_prob.size(0)):
        results.append({
            'category': categories[top5_catid[i]],
            'probability': top5_prob[i].item()
        })
    
    return results
```

### 2. Object Detection

Lokalizacja i klasyfikacja wielu obiektów na obrazie:

```python
import cv2
from ultralytics import YOLO

# YOLO v8
model = YOLO('yolov8n.pt')

def detect_objects(image_path):
    # Detekcja
    results = model(image_path)
    
    # Przetwarzanie wyników
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(confidence),
                'class': class_name
            })
    
    return detections

# Wizualizacja
def draw_detections(image_path, detections):
    image = cv2.imread(image_path)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class']}: {det['confidence']:.2f}"
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, label, (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    
    return image
```

### 3. Semantic Segmentation

Klasyfikacja każdego piksela:

```python
from torchvision.models.segmentation import deeplabv3_resnet101
import torch
import numpy as np

# Model
model = deeplabv3_resnet101(pretrained=True)
model.eval()

def segment_image(image):
    """
    Semantic segmentation
    
    Returns:
        Segmentation mask (H, W) gdzie każda wartość to class ID
    """
    # Preprocessing
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    # Argmax dla najlepszej klasy na pixel
    output_predictions = output.argmax(0).byte().cpu().numpy()
    
    return output_predictions

# Wizualizacja
def visualize_segmentation(image, segmentation):
    import matplotlib.pyplot as plt
    
    # Color map dla klas
    palette = np.array([
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        # ... więcej kolorów
    ])
    
    colored_mask = palette[segmentation]
    
    # Overlay
    overlay = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)
    
    return overlay
```

### 4. Instance Segmentation

Rozdzielenie poszczególnych instancji obiektów:

```python
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Konfiguracja Mask R-CNN
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

def instance_segmentation(image):
    """
    Detectron2 instance segmentation
    """
    outputs = predictor(image)
    
    instances = outputs["instances"].to("cpu")
    
    results = {
        'boxes': instances.pred_boxes.tensor.numpy(),
        'masks': instances.pred_masks.numpy(),
        'classes': instances.pred_classes.numpy(),
        'scores': instances.scores.numpy()
    }
    
    return results
```

## Depth Estimation

### Monocular Depth

```python
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image

# MiDaS / DPT model
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

def estimate_depth(image_path):
    image = Image.open(image_path)
    
    # Preprocessing
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Interpolate do oryginalnego rozmiaru
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    
    # Normalizacja
    depth = prediction.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    
    return depth
```

### Stereo Vision

```python
import cv2

class StereoDepth:
    def __init__(self):
        # Stereo matcher
        self.stereo = cv2.StereoBM_create(
            numDisparities=16*5,  # Must be divisible by 16
            blockSize=21
        )
    
    def compute_depth(self, left_image, right_image):
        """
        Oblicz depth map ze stereo pair
        """
        # Grayscale
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        
        # Disparity
        disparity = self.stereo.compute(left_gray, right_gray)
        
        # Convert to depth
        # depth = (baseline * focal_length) / disparity
        baseline = 0.1  # meters (distance between cameras)
        focal_length = 700  # pixels
        
        depth = np.zeros_like(disparity, dtype=np.float32)
        depth[disparity > 0] = (baseline * focal_length) / disparity[disparity > 0]
        
        return depth
```

## Optical Flow

Śledzenie ruchu pikseli między klatkami:

```python
def calculate_optical_flow(prev_gray, curr_gray):
    """
    Lucas-Kanade optical flow
    """
    # Parametry
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Detekcja feature points
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    
    p0 = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
    
    # Calculate flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, p0, None, **lk_params
    )
    
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    return good_old, good_new

# Wizualizacja
def draw_flow(image, old_points, new_points):
    for i, (new, old) in enumerate(zip(new_points, old_points)):
        a, b = new.ravel()
        c, d = old.ravel()
        
        # Rysuj linię ruchu
        cv2.line(image, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        cv2.circle(image, (int(a), int(b)), 5, (0, 0, 255), -1)
    
    return image
```

## Pose Estimation

### Human Pose

```python
import mediapipe as mp

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
    
    def estimate_pose(self, image):
        """
        Estymacja pozy człowieka (33 landmarks)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            return landmarks
        
        return None
    
    def draw_pose(self, image, landmarks):
        """
        Rysuj szkielet
        """
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        if landmarks:
            mp_drawing.draw_landmarks(
                image,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return image
```

### 6D Object Pose

```python
import cv2.aruco as aruco

class ObjectPoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # ArUco dictionary
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters_create()
    
    def estimate_pose(self, image, marker_size=0.05):
        """
        Estymacja 6D pose z ArUco markers
        
        Args:
            marker_size: Rozmiar markera w metrach
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detekcja markers
        corners, ids, rejected = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters
        )
        
        if ids is not None:
            # Estymacja pose
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, marker_size, 
                self.camera_matrix, self.dist_coeffs
            )
            
            poses = []
            for i in range(len(ids)):
                poses.append({
                    'id': ids[i][0],
                    'rotation': rvecs[i][0],
                    'translation': tvecs[i][0]
                })
            
            return poses
        
        return None
```

## Visual SLAM

```python
import cv2
import numpy as np

class VisualSLAM:
    def __init__(self):
        # ORB detector
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # FLANN matcher
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Camera parameters
        self.camera_matrix = None
        self.trajectory = []
    
    def process_frame(self, image):
        """
        Przetwórz frame i estymuj ruch kamery
        """
        # Detekcja features
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        
        if not hasattr(self, 'prev_keypoints'):
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return None
        
        # Matching
        matches = self.matcher.knnMatch(
            self.prev_descriptors, descriptors, k=2
        )
        
        # Ratio test (Lowe's)
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) > 10:
            # Ekstract matched points
            src_pts = np.float32([
                self.prev_keypoints[m.queryIdx].pt for m in good_matches
            ]).reshape(-1, 1, 2)
            
            dst_pts = np.float32([
                keypoints[m.trainIdx].pt for m in good_matches
            ]).reshape(-1, 1, 2)
            
            # Essential matrix
            E, mask = cv2.findEssentialMat(
                src_pts, dst_pts, 
                self.camera_matrix, method=cv2.RANSAC
            )
            
            # Recover pose
            _, R, t, mask = cv2.recoverPose(
                E, src_pts, dst_pts, self.camera_matrix
            )
            
            # Update trajectory
            if len(self.trajectory) == 0:
                self.trajectory.append(np.eye(4))
            else:
                # Compose transformation
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t.flatten()
                
                self.trajectory.append(self.trajectory[-1] @ T)
        
        # Update previous
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        return self.trajectory
```

## Integracja z ROS2

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D
from cv_bridge import CvBridge

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        self.bridge = CvBridge()
        self.detector = YOLO('yolov8n.pt')
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publisher
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            '/vision/detections',
            10
        )
    
    def image_callback(self, msg):
        # Konwersja
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Detekcja
        results = self.detector(cv_image)
        
        # Publikacja
        detection_array = Detection2DArray()
        detection_array.header = msg.header
        
        for result in results:
            for box in result.boxes:
                det = Detection2D()
                # ... wypełnij detection
                detection_array.detections.append(det)
        
        self.detections_pub.publish(detection_array)
```

## Powiązane Artykuły

- [Deep Learning](#wiki-deep-learning)
- [Detekcja Twarzy](#wiki-face-detection)
- [OpenCV](#wiki-opencv)
- [MediaPipe](#wiki-mediapipe)

---

*Ostatnia aktualizacja: 2025-02-10*  
*Autor: Zespół Percepcji, Laboratorium Robotów Humanoidalnych*
