# Detekcja Twarzy (Face Detection)

## Wprowadzenie

**Detekcja twarzy** to proces lokalizacji i identyfikacji twarzy ludzkich na obrazach lub w sekwencjach wideo. Jest to fundamentalny krok dla większości aplikacji związanych z analizą twarzy, rozpoznawaniem emocji i biometrią.

## Metody Detekcji

### 1. Klasyczne Metody (Pre-Deep Learning)

#### Viola-Jones Algorithm (2001)

Pierwszy efektywny algorytm real-time:

```python
import cv2

# Wczytaj klasyfikator Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Detekcja
image = cv2.imread('photo.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Rysowanie prostokątów
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
```

**Zalety:**
- Bardzo szybki (real-time na CPU)
- Niskie wymagania obliczeniowe

**Wady:**
- Słaba dokładność przy różnych kątach
- Wysokie False Positive Rate
- Problemy z oświetleniem

### 2. Deep Learning Methods

#### MTCNN (Multi-task Cascaded CNN)

```python
from facenet_pytorch import MTCNN
import torch
from PIL import Image

# Inicjalizacja detektora
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(
    keep_all=True,
    device=device,
    post_process=False,
    min_face_size=20
)

# Detekcja
image = Image.open('photo.jpg')
boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)

# Wizualizacja
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image)

if boxes is not None:
    for box, prob in zip(boxes, probs):
        if prob > 0.9:  # Confidence threshold
            rect = Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor='g',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                box[0], box[1] - 10,
                f'{prob:.2f}',
                color='white',
                fontsize=12,
                bbox=dict(facecolor='green', alpha=0.5)
            )

plt.axis('off')
plt.show()
```

**Zalety:**
- Wysoka dokładność
- Detekcja face landmarks (5 punktów)
- Dobra wydajność

**Wady:**
- Wymaga GPU dla real-time
- Większe zużycie pamięci

#### RetinaFace

State-of-the-art detektor (2020):

```python
from retinaface import RetinaFace

# Detekcja
faces = RetinaFace.detect_faces('photo.jpg')

# Analiza wyników
for key in faces.keys():
    identity = faces[key]
    
    # Bounding box
    facial_area = identity["facial_area"]
    x, y, w, h = facial_area
    
    # Landmarks (5 punktów)
    landmarks = identity["landmarks"]
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    nose = landmarks["nose"]
    mouth_left = landmarks["mouth_left"]
    mouth_right = landmarks["mouth_right"]
    
    # Confidence
    score = identity["score"]
    
    print(f"Face {key}: confidence={score:.3f}, bbox={facial_area}")
```

### 3. MediaPipe Face Detection

Google's solution z face mesh:

```python
import mediapipe as mp
import cv2

# Inicjalizacja
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Detektor
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  # 0: krótki zasięg, 1: długi zasięg
    min_detection_confidence=0.5
)

# Przetwarzanie wideo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Konwersja BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detekcja
    results = face_detection.process(image_rgb)
    
    # Rysowanie
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)
    
    cv2.imshow('MediaPipe Face Detection', image)
    
    if cv2.waitKey(5) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
```

## Face Landmarks Detection

### 68-Point Facial Landmarks (dlib)

```python
import dlib
import cv2
import numpy as np

# Detektor twarzy
detector = dlib.get_frontal_face_detector()

# Predictor landmarks
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load image
image = cv2.imread('face.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detekcja twarzy
faces = detector(gray)

# Dla każdej twarzy
for face in faces:
    # Predykcja landmarks
    landmarks = predictor(gray, face)
    
    # Rysowanie punktów
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

cv2.imshow('68 Landmarks', image)
cv2.waitKey(0)
```

### 478-Point Face Mesh (MediaPipe)

```python
import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,  # 478 punktów
    min_detection_confidence=0.5
)

# Load image
image = cv2.imread('face.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process
results = face_mesh.process(image_rgb)

# Rysowanie
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
        )

cv2.imshow('Face Mesh', image)
cv2.waitKey(0)
```

## Face Alignment

Normalizacja orientacji twarzy:

```python
import cv2
import numpy as np

def align_face(image, landmarks):
    """
    Wyrównanie twarzy na podstawie landmarks oczu
    """
    # Pobierz pozycje oczu (landmarks 36-41 i 42-47 dla dlib 68-point)
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    
    # Oblicz kąt
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Środek między oczami
    eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                   (left_eye[1] + right_eye[1]) / 2)
    
    # Macierz rotacji
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    
    # Zastosuj transformację
    aligned = cv2.warpAffine(
        image, M, 
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC
    )
    
    return aligned

# Użycie
aligned_face = align_face(image, landmarks)
```

## Face Tracking w Wideo

```python
import cv2
from collections import deque

class FaceTracker:
    def __init__(self, max_track_history=30):
        self.detector = MTCNN()
        self.tracks = {}
        self.next_id = 0
        self.max_history = max_track_history
    
    def update(self, frame):
        # Detekcja
        boxes, probs = self.detector.detect(frame)
        
        if boxes is None:
            return self.tracks
        
        # Matching z poprzednimi track-ami
        new_tracks = {}
        used_boxes = set()
        
        for track_id, track_data in self.tracks.items():
            last_box = track_data['history'][-1]
            
            # Znajdź najbliższy box
            min_dist = float('inf')
            best_idx = -1
            
            for idx, box in enumerate(boxes):
                if idx in used_boxes:
                    continue
                
                dist = self.box_distance(last_box, box)
                if dist < min_dist and dist < 50:  # threshold
                    min_dist = dist
                    best_idx = idx
            
            if best_idx != -1:
                # Update track
                track_data['history'].append(boxes[best_idx])
                if len(track_data['history']) > self.max_history:
                    track_data['history'].popleft()
                
                new_tracks[track_id] = track_data
                used_boxes.add(best_idx)
        
        # Nowe twarze
        for idx, box in enumerate(boxes):
            if idx not in used_boxes:
                new_tracks[self.next_id] = {
                    'history': deque([box], maxlen=self.max_history)
                }
                self.next_id += 1
        
        self.tracks = new_tracks
        return self.tracks
    
    def box_distance(self, box1, box2):
        # Euclidean distance między centrami boxów
        c1 = np.array([(box1[0] + box1[2])/2, (box1[1] + box1[3])/2])
        c2 = np.array([(box2[0] + box2[2])/2, (box2[1] + box2[3])/2])
        return np.linalg.norm(c1 - c2)

# Użycie
tracker = FaceTracker()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    tracks = tracker.update(frame)
    
    # Rysowanie
    for track_id, data in tracks.items():
        box = data['history'][-1]
        cv2.rectangle(frame, 
                     (int(box[0]), int(box[1])),
                     (int(box[2]), int(box[3])),
                     (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}',
                   (int(box[0]), int(box[1])-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Face Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

## Optymalizacja Wydajności

### Batch Processing

```python
def batch_detect(images, batch_size=32):
    """
    Batch detection dla wielu obrazów
    """
    detector = MTCNN(keep_all=True)
    
    all_results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        results = detector.detect(batch)
        all_results.extend(results)
    
    return all_results
```

### Cascade Detection

```python
def cascade_detect(image):
    """
    Najpierw szybki detektor, potem dokładny
    """
    # Szybka pre-detekcja (Haar Cascade)
    fast_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    quick_faces = fast_detector.detectMultiScale(gray)
    
    if len(quick_faces) == 0:
        return []
    
    # Dokładna detekcja (MTCNN) tylko w regionach
    detailed_faces = []
    for (x, y, w, h) in quick_faces:
        # ROI z marginesem
        margin = 20
        roi = image[
            max(0, y-margin):min(image.shape[0], y+h+margin),
            max(0, x-margin):min(image.shape[1], x+w+margin)
        ]
        
        mtcnn = MTCNN()
        boxes, _ = mtcnn.detect(roi)
        
        if boxes is not None:
            # Adjust coordinates
            for box in boxes:
                box[0] += x - margin
                box[1] += y - margin
                box[2] += x - margin
                box[3] += y - margin
                detailed_faces.append(box)
    
    return detailed_faces
```

## Porównanie Metod

| Metoda | FPS (CPU) | FPS (GPU) | Accuracy | Landmarks |
|--------|-----------|-----------|----------|-----------|
| Haar Cascade | 30+ | N/A | ★★☆☆☆ | ✗ |
| MTCNN | 5-10 | 30-60 | ★★★★☆ | 5 pts |
| RetinaFace | 2-5 | 20-40 | ★★★★★ | 5 pts |
| MediaPipe | 15-25 | 60+ | ★★★★☆ | 6 pts |
| dlib | 10-15 | N/A | ★★★★☆ | 68 pts |

## Zastosowania w Robotyce

```python
# Integracja z ROS2
class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection')
        
        self.detector = MTCNN()
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.faces_pub = self.create_publisher(
            FaceArray,
            '/perception/faces',
            10
        )
    
    def image_callback(self, msg):
        # Konwersja
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Detekcja
        boxes, probs = self.detector.detect(cv_image)
        
        if boxes is not None:
            # Publikacja
            face_array = FaceArray()
            face_array.header = msg.header
            
            for box, prob in zip(boxes, probs):
                face = Face()
                face.bbox = [int(x) for x in box]
                face.confidence = float(prob)
                face_array.faces.append(face)
            
            self.faces_pub.publish(face_array)
```

## Powiązane Artykuły

- [Rozpoznawanie Emocji](#wiki-emotion-recognition)
- [Informatyka Afektywna](#wiki-affective-computing)
- [DeepFace](#wiki-deepface)

---

*Ostatnia aktualizacja: 2025-02-10*  
*Autor: Zespół Percepcji, Laboratorium Robotów Humanoidalnych*
