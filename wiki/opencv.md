# OpenCV - Open Source Computer Vision Library

## Wprowadzenie

**OpenCV** (Open Source Computer Vision Library) to najpopularniejsza biblioteka do przetwarzania obrazów i wizji komputerowej. W Laboratorium używana jest do wszystkich zadań związanych z percepcją wizualną robota Unitree G1.

## Instalacja

```bash
# Python
pip install opencv-python opencv-contrib-python

# Z CUDA (GPU support)
pip install opencv-contrib-python-headless
```

## Podstawowe Operacje

### Wczytywanie i Wyświetlanie

```python
import cv2
import numpy as np

# Wczytaj obraz
image = cv2.imread('robot.jpg')

# Wyświetl
cv2.imshow('Robot', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Info
print(f"Shape: {image.shape}")  # (height, width, channels)
print(f"Type: {image.dtype}")   # uint8

# Save
cv2.imwrite('output.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
```

### Konwersje Kolorów

```python
# BGR -> RGB
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# BGR -> Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# BGR -> HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Grayscale -> BGR
bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
```

### Operacje Geometryczne

```python
# Resize
resized = cv2.resize(image, (640, 480))
resized = cv2.resize(image, None, fx=0.5, fy=0.5)  # 50% scale

# Rotate
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)  # 45 degrees
rotated = cv2.warpAffine(image, M, (w, h))

# Flip
flipped_h = cv2.flip(image, 1)  # Horizontal
flipped_v = cv2.flip(image, 0)  # Vertical
flipped_hv = cv2.flip(image, -1)  # Both

# Crop
cropped = image[100:400, 200:500]  # [y1:y2, x1:x2]
```

## Filtrowanie

### Blur

```python
# Gaussian Blur
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Median Blur (good for salt-and-pepper noise)
median = cv2.medianBlur(image, 5)

# Bilateral Filter (edge-preserving)
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
```

### Detekcja Krawędzi

```python
# Canny
edges = cv2.Canny(gray, 50, 150)

# Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

# Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
```

## Detekcja Obiektów

### Template Matching

```python
def find_template(image, template):
    """
    Find template in image
    """
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    
    threshold = 0.8
    loc = np.where(result >= threshold)
    
    matches = []
    for pt in zip(*loc[::-1]):
        matches.append({
            'top_left': pt,
            'bottom_right': (pt[0] + template.shape[1], pt[1] + template.shape[0])
        })
    
    return matches

# Użycie
template = cv2.imread('object.jpg', 0)
image = cv2.imread('scene.jpg', 0)
matches = find_template(image, template)

# Rysuj
for match in matches:
    cv2.rectangle(image, match['top_left'], match['bottom_right'], (0, 255, 0), 2)
```

### Contour Detection

```python
def detect_objects(image):
    """
    Detect objects using contours
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area > 100:  # Filter small contours
            x, y, w, h = cv2.boundingRect(cnt)
            objects.append({
                'bbox': (x, y, w, h),
                'area': area,
                'contour': cnt
            })
    
    return objects

# Rysowanie
objects = detect_objects(image)
for obj in objects:
    x, y, w, h = obj['bbox']
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.drawContours(image, [obj['contour']], -1, (0, 0, 255), 2)
```

### Hough Transform

```python
# Line Detection
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLinesP(
    edges, 1, np.pi/180, 100,
    minLineLength=100,
    maxLineGap=10
)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Circle Detection
circles = cv2.HoughCircles(
    gray, cv2.HOUGH_GRADIENT, 1, 20,
    param1=50, param2=30,
    minRadius=0, maxRadius=0
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Outer circle
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)  # Center
```

## Feature Detection

### ORB (Oriented FAST and Rotated BRIEF)

```python
# Detector
orb = cv2.ORB_create(nfeatures=1000)

# Detect and compute
keypoints, descriptors = orb.detectAndCompute(gray, None)

# Draw keypoints
image_with_kp = cv2.drawKeypoints(
    image, keypoints, None, 
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
```

### SIFT (Scale-Invariant Feature Transform)

```python
# SIFT
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Feature Matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(desc1, desc2)

# Sort by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
matched_image = cv2.drawMatches(
    img1, kp1, img2, kp2, 
    matches[:10], None, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
```

## Calibracja Kamery

```python
def calibrate_camera(images, pattern_size=(9, 6)):
    """
    Calibrate camera using chessboard
    
    Args:
        images: List of calibration images
        pattern_size: Chessboard inner corners (width, height)
    """
    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    objpoints = []  # 3D points
    imgpoints = []  # 2D points
    
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            objpoints.append(objp)
            
            # Refine corners
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners2)
    
    # Calibrate
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    return camera_matrix, dist_coeffs

# Undistort
def undistort_image(image, camera_matrix, dist_coeffs):
    h, w = image.shape[:2]
    newcamera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    undistorted = cv2.undistort(
        image, camera_matrix, dist_coeffs, None, newcamera_matrix
    )
    
    return undistorted
```

## Video Processing

```python
class VideoProcessor:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    
    def process_frame(self, frame):
        """Override this method"""
        return frame
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process
            processed = self.process_frame(frame)
            
            # Display
            cv2.imshow('Video', processed)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

# Recording
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        out.write(frame)

out.release()
```

## ROS2 Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class OpenCVNode(Node):
    def __init__(self):
        super().__init__('opencv_node')
        
        self.bridge = CvBridge()
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publisher
        self.processed_pub = self.create_publisher(
            Image,
            '/camera/processed',
            10
        )
    
    def image_callback(self, msg):
        # ROS Image -> OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Process
        processed = self.process_image(cv_image)
        
        # OpenCV -> ROS Image
        processed_msg = self.bridge.cv2_to_imgmsg(processed, 'bgr8')
        processed_msg.header = msg.header
        
        # Publish
        self.processed_pub.publish(processed_msg)
    
    def process_image(self, image):
        # Edge detection example
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def main(args=None):
    rclpy.init(args=args)
    node = OpenCVNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Performance Tips

```python
# Use numpy operations when possible
result = cv2.add(img1, img2)  # Better than img1 + img2

# ROI (Region of Interest)
roi = image[y:y+h, x:x+w]
roi_processed = cv2.GaussianBlur(roi, (5, 5), 0)
image[y:y+h, x:x+w] = roi_processed

# GPU acceleration (if available)
import cv2.cuda as cuda

gpu_frame = cuda.GpuMat()
gpu_frame.upload(frame)
gpu_result = cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
result = gpu_result.download()
```

## Powiązane Artykuły

- [Computer Vision](#wiki-computer-vision)
- [Detekcja Obiektów](#wiki-object-detection)
- [MediaPipe](#wiki-mediapipe)
- [Deep Learning](#wiki-deep-learning)

## Zasoby

- [OpenCV Documentation](https://docs.opencv.org/)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- [OpenCV Python Tutorial](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

---

*Ostatnia aktualizacja: 2025-02-10*  
*Autor: Zespół Percepcji, Laboratorium Robotów Humanoidalnych*
