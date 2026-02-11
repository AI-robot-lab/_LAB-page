# DeepFace - Framework do Analizy Twarzy

## Wprowadzenie

**DeepFace** to lekki framework Pythonowy do analizy twarzy, łączący w sobie różne modele deep learning. Stworzony przez Sefika Ilkina Serengila, oferuje ujednolicone API dla:
- Weryfikacji twarzy (Face Recognition)
- Rozpoznawania emocji
- Detekcji wieku i płci
- Rozpoznawania rasy/etniczności

## Instalacja

```bash
pip install deepface
```

Zależności:
- TensorFlow 2.x lub PyTorch
- OpenCV
- Keras

## Podstawowe Użycie

### 1. Face Recognition

```python
from deepface import DeepFace

# Weryfikacja czy dwie twarze to ta sama osoba
result = DeepFace.verify(
    img1_path="person1.jpg",
    img2_path="person2.jpg",
    model_name="VGG-Face"  # lub Facenet, ArcFace, etc.
)

print(f"Ta sama osoba: {result['verified']}")
print(f"Distance: {result['distance']}")
print(f"Threshold: {result['threshold']}")
```

### 2. Face Analysis (All-in-One)

```python
# Kompleksowa analiza twarzy
result = DeepFace.analyze(
    img_path="face.jpg",
    actions=['age', 'gender', 'race', 'emotion'],
    enforce_detection=True
)

# Wyniki
print(f"Wiek: {result[0]['age']}")
print(f"Płeć: {result[0]['dominant_gender']}")
print(f"Emocja: {result[0]['dominant_emotion']}")
print(f"Emocje: {result[0]['emotion']}")
print(f"Rasa: {result[0]['dominant_race']}")
```

### 3. Face Detection

```python
# Detekcja twarzy z różnymi backendami
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

for backend in backends:
    face_objs = DeepFace.extract_faces(
        img_path="group.jpg",
        detector_backend=backend
    )
    
    print(f"{backend}: {len(face_objs)} twarzy wykrytych")
```

## Dostępne Modele

### Face Recognition Models

| Model | Rozmiar | Accuracy | Szybkość |
|-------|---------|----------|----------|
| **VGG-Face** | 515 MB | ★★★★☆ | Średnia |
| **Facenet** | 23 MB | ★★★★★ | Szybka |
| **Facenet512** | 95 MB | ★★★★★ | Szybka |
| **OpenFace** | 25 MB | ★★★☆☆ | Bardzo szybka |
| **DeepFace** | 152 MB | ★★★★☆ | Wolna |
| **DeepID** | 29 MB | ★★★☆☆ | Szybka |
| **ArcFace** | 166 MB | ★★★★★ | Średnia |
| **Dlib** | 22 MB | ★★★☆☆ | Szybka |
| **SFace** | 34 MB | ★★★★☆ | Szybka |

### Detector Backends

- **opencv** - Haar Cascade (najszybszy, najmniej dokładny)
- **ssd** - Single Shot Detector (dobry kompromis)
- **dlib** - HOG (dobry dla frontalnych twarzy)
- **mtcnn** - Multi-task CNN (bardzo dokładny)
- **retinaface** - State-of-the-art (najdokładniejszy)
- **mediapipe** - Google's solution (szybki i dokładny)
- **yolov8** - Latest YOLO (bardzo szybki)
- **yunet** - CNN-based (dobry kompromis)
- **fastmtcnn** - Zoptymalizowany MTCNN

## Zaawansowane Użycie

### Batch Processing

```python
import os
from deepface import DeepFace

def batch_analyze(image_folder):
    """
    Analiza wielu obrazów
    """
    results = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            filepath = os.path.join(image_folder, filename)
            
            try:
                result = DeepFace.analyze(
                    img_path=filepath,
                    actions=['emotion', 'age', 'gender'],
                    enforce_detection=False,
                    silent=True
                )
                
                results.append({
                    'file': filename,
                    'emotion': result[0]['dominant_emotion'],
                    'age': result[0]['age'],
                    'gender': result[0]['dominant_gender']
                })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return results

# Użycie
results = batch_analyze('images/')
for r in results:
    print(f"{r['file']}: {r['emotion']}, {r['age']} lat, {r['gender']}")
```

### Real-time Analiza Wideo

```python
import cv2
from deepface import DeepFace

def realtime_emotion_detection():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # Analiza co N-tą klatkę (dla wydajności)
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Rysowanie wyniku
            emotion = result[0]['dominant_emotion']
            cv2.putText(
                frame,
                f"Emotion: {emotion}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
        except Exception as e:
            pass
        
        cv2.imshow('Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Uruchom
realtime_emotion_detection()
```

### Face Database Search

```python
from deepface import DeepFace
import pandas as pd

# Zbuduj bazę danych twarzy
def build_face_database(database_path):
    """
    Reprezentacja wszystkich twarzy w folderze
    """
    DeepFace.find(
        img_path="target.jpg",
        db_path=database_path,
        model_name="VGG-Face",
        enforce_detection=False,
        silent=True
    )
    print("Database built successfully!")

# Wyszukiwanie podobnych twarzy
def find_similar_faces(query_image, database_path):
    dfs = DeepFace.find(
        img_path=query_image,
        db_path=database_path,
        model_name="Facenet512",
        distance_metric="cosine",
        enforce_detection=False
    )
    
    if len(dfs) > 0 and len(dfs[0]) > 0:
        # Top 5 podobnych twarzy
        top_matches = dfs[0].head(5)
        print("Top 5 podobnych twarzy:")
        for idx, row in top_matches.iterrows():
            print(f"{row['identity']}: distance={row['distance']:.4f}")
    else:
        print("Nie znaleziono podobnych twarzy")
    
    return dfs

# Użycie
database_path = "face_database/"
query_image = "person_to_find.jpg"

results = find_similar_faces(query_image, database_path)
```

## Custom Model Training

### Własny Model Emocji

```python
from tensorflow import keras
from deepface.commons import functions
import numpy as np

class CustomEmotionModel:
    def __init__(self):
        # Załaduj pre-trained model
        self.base_model = keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base layers
        for layer in self.base_model.layers:
            layer.trainable = False
        
        # Custom head
        x = keras.layers.Flatten()(self.base_model.output)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        output = keras.layers.Dense(7, activation='softmax')(x)  # 7 emocji
        
        self.model = keras.Model(
            inputs=self.base_model.input,
            outputs=output
        )
    
    def train(self, X_train, y_train, epochs=50):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_split=0.2
        )
        
        return history
    
    def predict(self, image):
        # Preprocessing
        img = functions.preprocess_face(
            img=image,
            target_size=(224, 224),
            enforce_detection=False
        )
        
        # Predykcja
        prediction = self.model.predict(np.expand_dims(img, axis=0))
        
        emotions = ['angry', 'disgust', 'fear', 'happy', 
                   'sad', 'surprise', 'neutral']
        
        result = {
            emotion: float(prob) 
            for emotion, prob in zip(emotions, prediction[0])
        }
        
        dominant = emotions[np.argmax(prediction)]
        
        return {
            'emotion': result,
            'dominant_emotion': dominant
        }
```

## Integracja z ROS2

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from deepface import DeepFace
import json

class DeepFaceNode(Node):
    def __init__(self):
        super().__init__('deepface_node')
        
        self.bridge = CvBridge()
        
        # Parametry
        self.declare_parameter('model_name', 'VGG-Face')
        self.declare_parameter('detector_backend', 'retinaface')
        self.declare_parameter('analyze_rate', 2.0)  # Hz
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publishers
        self.emotion_pub = self.create_publisher(
            String,
            '/perception/emotion',
            10
        )
        
        self.analysis_pub = self.create_publisher(
            String,
            '/perception/face_analysis',
            10
        )
        
        # Timer dla rate limiting
        self.timer = self.create_timer(
            1.0 / self.get_parameter('analyze_rate').value,
            self.analyze_timer_callback
        )
        
        self.latest_frame = None
        
        self.get_logger().info('DeepFace Node started')
    
    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    
    def analyze_timer_callback(self):
        if self.latest_frame is None:
            return
        
        try:
            result = DeepFace.analyze(
                self.latest_frame,
                actions=['emotion', 'age', 'gender'],
                enforce_detection=False,
                detector_backend=self.get_parameter('detector_backend').value,
                silent=True
            )
            
            if len(result) > 0:
                # Publikacja emocji
                emotion_msg = String()
                emotion_msg.data = result[0]['dominant_emotion']
                self.emotion_pub.publish(emotion_msg)
                
                # Publikacja pełnej analizy
                analysis_msg = String()
                analysis_msg.data = json.dumps({
                    'emotion': result[0]['dominant_emotion'],
                    'emotion_scores': result[0]['emotion'],
                    'age': result[0]['age'],
                    'gender': result[0]['dominant_gender']
                })
                self.analysis_pub.publish(analysis_msg)
                
        except Exception as e:
            self.get_logger().warn(f'Analysis failed: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = DeepFaceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Optymalizacja Wydajności

### GPU Acceleration

```python
import tensorflow as tf

# Sprawdź GPU
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Konfiguracja GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Użyj DeepFace normalnie - automatycznie użyje GPU
result = DeepFace.verify("img1.jpg", "img2.jpg")
```

### Model Caching

```python
from deepface import DeepFace
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model(model_name):
    """
    Cache modelu w pamięci
    """
    # DeepFace automatycznie cache'uje modele
    return True

# Pierwsze wywołanie ładuje model
DeepFace.verify("img1.jpg", "img2.jpg", model_name="VGG-Face")

# Kolejne wywołania używają cached model
DeepFace.verify("img3.jpg", "img4.jpg", model_name="VGG-Face")
```

## Porównanie z Innymi Bibliotekami

| Feature | DeepFace | face_recognition | InsightFace |
|---------|----------|------------------|-------------|
| Modele | 9 | 1 | 5+ |
| Emocje | ✓ | ✗ | ✗ |
| Wiek/Płeć | ✓ | ✗ | ✓ |
| Łatwość użycia | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| Wydajność | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| Dokumentacja | ★★★★☆ | ★★★★★ | ★★★☆☆ |

## Ograniczenia

1. **Wydajność Real-time**
   - Wolniejsze niż dedykowane modele
   - Wymaga GPU dla smooth video processing

2. **Dokładność Detekcji**
   - Zależna od wybranego backend
   - Może mieć problemy z profil i częściowa okludacja

3. **Model Size**
   - Niektóre modele są duże (>500MB)
   - Może być problem na urządzeniach edge

## Best Practices

```python
# 1. Użyj enforce_detection=False dla real-time
result = DeepFace.analyze(
    frame,
    enforce_detection=False,  # Nie rzuca wyjątku jeśli brak twarzy
    silent=True  # Suppress warnings
)

# 2. Wybierz odpowiedni model dla use case
# - Szybkość: OpenFace, SFace
# - Accuracy: Facenet512, ArcFace
# - Balans: VGG-Face

# 3. Użyj batch processing dla wielu obrazów
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
for img in images:
    result = DeepFace.analyze(img, silent=True)

# 4. Cache representations dla face search
DeepFace.find(
    img_path="query.jpg",
    db_path="database/",
    model_name="Facenet512",
    enforce_detection=False
)
```

## Powiązane Artykuły

- [Detekcja Twarzy](#wiki-face-detection)
- [Rozpoznawanie Emocji](#wiki-emotion-recognition)
- [Informatyka Afektywna](#wiki-affective-computing)

## Zasoby

- [GitHub](https://github.com/serengil/deepface)
- [Dokumentacja](https://github.com/serengil/deepface/blob/master/README.md)
- [PyPI](https://pypi.org/project/deepface/)

---

*Ostatnia aktualizacja: 2025-02-10*  
*Autor: Zespół Percepcji, Laboratorium Robotów Humanoidalnych*
