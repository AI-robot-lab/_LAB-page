# Object Detection - Detekcja Obiektów

## Wprowadzenie

**Object Detection** to zadanie lokalizacji i klasyfikacji obiektów na obrazach. W robotyce humanoidalnej umożliwia rozpoznawanie i manipulację przedmiotami w otoczeniu.

## YOLO (You Only Look Once)

### YOLOv8 z Ultralytics

```python
from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        
        # Class names (COCO dataset)
        self.classes = self.model.names
    
    def detect(self, image, conf_threshold=0.5):
        """
        Detect objects in image
        
        Returns:
            List of detections with bbox, class, confidence
        """
        results = self.model(image, conf=conf_threshold)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extract data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                cls = int(box.cls[0].cpu().item())
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class': self.classes[cls],
                    'confidence': conf
                })
        
        return detections
    
    def draw_detections(self, image, detections):
        """
        Draw bounding boxes on image
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']} {det['confidence']:.2f}"
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image

# Usage
detector = ObjectDetector('yolov8n.pt')
image = cv2.imread('scene.jpg')

detections = detector.detect(image, conf_threshold=0.5)
image_with_boxes = detector.draw_detections(image, detections)

cv2.imshow('Detections', image_with_boxes)
cv2.waitKey(0)
```

## Faster R-CNN

```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class FasterRCNNDetector:
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # COCO classes
        self.classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def detect(self, image, threshold=0.5):
        """
        Detect objects using Faster R-CNN
        """
        # Preprocess
        image_tensor = torchvision.transforms.ToTensor()(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Extract detections
        pred = predictions[0]
        
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        # Filter by threshold
        mask = scores > threshold
        
        detections = []
        for box, label, score in zip(boxes[mask], labels[mask], scores[mask]):
            detections.append({
                'bbox': box.astype(int).tolist(),
                'class': self.classes[label],
                'confidence': float(score)
            })
        
        return detections
```

## Training Custom Detector

```python
from ultralytics import YOLO

def train_custom_yolo():
    """
    Train YOLOv8 on custom dataset
    """
    # Load pretrained model
    model = YOLO('yolov8n.pt')
    
    # Train
    results = model.train(
        data='custom_data.yaml',  # Dataset config
        epochs=100,
        imgsz=640,
        batch=16,
        name='custom_detector'
    )
    
    # Validate
    metrics = model.val()
    
    return model

# Dataset YAML format:
"""
# custom_data.yaml
path: /path/to/dataset
train: images/train
val: images/val

names:
  0: cup
  1: bottle
  2: plate
  3: fork
"""
```

## Instance Segmentation

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn

class MaskRCNN:
    def __init__(self):
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def segment(self, image, threshold=0.5):
        """
        Instance segmentation
        
        Returns masks for each detected object
        """
        image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        pred = predictions[0]
        
        # Extract masks
        masks = pred['masks'].cpu().numpy()
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        # Filter
        mask = scores > threshold
        
        results = []
        for m, box, label, score in zip(masks[mask], boxes[mask], labels[mask], scores[mask]):
            results.append({
                'mask': (m[0] > 0.5).astype('uint8'),
                'bbox': box.astype(int).tolist(),
                'class': label,
                'confidence': float(score)
            })
        
        return results
```

## ROS2 Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector')
        
        self.detector = ObjectDetector('yolov8n.pt')
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/detections',
            10
        )
    
    def image_callback(self, msg):
        # Convert to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Detect
        detections = self.detector.detect(cv_image)
        
        # Publish
        self.publish_detections(detections)
    
    def publish_detections(self, detections):
        msg = Detection2DArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        
        for det in detections:
            detection = Detection2D()
            
            # Bounding box
            x1, y1, x2, y2 = det['bbox']
            detection.bbox.center.x = float((x1 + x2) / 2)
            detection.bbox.center.y = float((y1 + y2) / 2)
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)
            
            # Hypothesis
            hyp = ObjectHypothesisWithPose()
            hyp.id = det['class']
            hyp.score = det['confidence']
            detection.results.append(hyp)
            
            msg.detections.append(detection)
        
        self.detection_pub.publish(msg)
```

## Porównanie Modeli

| Model | FPS (GPU) | mAP | Parametry | Use Case |
|-------|-----------|-----|-----------|----------|
| **YOLOv8n** | 140 | 37.3 | 3.2M | Real-time |
| **YOLOv8m** | 85 | 50.2 | 25.9M | Balanced |
| **Faster R-CNN** | 15 | 42.0 | 41.8M | Accuracy |
| **RetinaNet** | 20 | 39.1 | 36.3M | Dense objects |
| **DETR** | 28 | 42.0 | 41.3M | Transformer |

## Powiązane Artykuły

- [Computer Vision](#wiki-computer-vision)
- [Deep Learning](#wiki-deep-learning)
- [PyTorch](#wiki-pytorch)

---

*Ostatnia aktualizacja: 2025-02-11*
