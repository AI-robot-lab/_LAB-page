# Informatyka Afektywna

## Definicja

**Informatyka afektywna** (ang. *Affective Computing*) to dziedzina nauki zajmująca się rozpoznawaniem, interpretacją, przetwarzaniem i symulowaniem stanów emocjonalnych człowieka przez systemy komputerowe.

Termin został wprowadzony przez **Rosalind Picard** z MIT w 1995 roku.

## Znaczenie w Robotyce Humanoidalnej

W kontekście robotów humanoidalnych, informatyka afektywna umożliwia:
- **Naturalne interakcje** - robot rozumie emocje użytkownika
- **Adaptacyjne zachowanie** - dostosowanie reakcji do stanu emocjonalnego
- **Empatyczne wsparcie** - szczególnie ważne w rehabilitacji
- **Wykrywanie stresu** - monitoring zdrowia psychicznego

## Modalności Percepcji Emocji

### 1. Analiza Twarzy

Rozpoznawanie emocji na podstawie ekspresji twarzy (Facial Action Coding System - FACS):

**Podstawowe emocje (Paul Ekman):**
- Radość (happiness)
- Smutek (sadness)
- Gniew (anger)
- Strach (fear)
- Zaskoczenie (surprise)
- Wstręt (disgust)

```python
from deepface import DeepFace

# Analiza emocji z obrazu
result = DeepFace.analyze(
    img_path="face.jpg",
    actions=['emotion'],
    enforce_detection=False
)

print(f"Dominująca emocja: {result[0]['dominant_emotion']}")
print(f"Rozkład emocji: {result[0]['emotion']}")
```

### 2. Analiza Głosu

Parametry prozodyczne:
- Wysokość tonu (pitch)
- Tempo mowy
- Intensywność
- Jitter i shimmer
- Pauzy w mowie

### 3. Sygnały Fizjologiczne

- **HR** - częstość akcji serca
- **GSR** - przewodność skóry (Galvanic Skin Response)
- **EEG** - aktywność mózgu
- **EMG** - napięcie mięśni

### 4. Kontekst Sytuacyjny

- Analiza mowy ciała (body language)
- Śledzenie wzroku (eye tracking)
- Proxemics - dystans przestrzenny

## Modele Emocji

### Model Dyskretny (Ekman)

6 podstawowych emocji uniwersalnych dla wszystkich kultur.

### Model Wymiarowy (Russell)

**Circumplex Model of Affect:**
- Oś Walencji (Valence): negatywna ↔ pozytywna
- Oś Pobudzenia (Arousal): niskie ↔ wysokie

```
      Wysoki Arousal
            ↑
Negatywny ←─┼─→ Pozytywny
     Valence │
            ↓
      Niski Arousal
```

### Model Komponentowy (Scherer)

Emocje jako kombinacja:
- Oceny poznawczej (appraisal)
- Pobudzenia fizjologicznego
- Ekspresji ruchowej
- Gotowości do działania
- Subiektywnego odczucia

## Implementacja w Laboratorium

### Pipeline Percepcji Afektywnej

```python
import cv2
from deepface import DeepFace
import mediapipe as mp

class AffectivePerception:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def analyze_frame(self, frame):
        # Detekcja twarzy i landmarks
        results = self.face_mesh.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        
        # Analiza emocji
        try:
            emotion_analysis = DeepFace.analyze(
                frame,
                actions=['emotion', 'age', 'gender'],
                enforce_detection=False,
                silent=True
            )
            
            return {
                'face_detected': results.multi_face_landmarks is not None,
                'emotion': emotion_analysis[0]['dominant_emotion'],
                'emotion_scores': emotion_analysis[0]['emotion'],
                'age': emotion_analysis[0]['age'],
                'gender': emotion_analysis[0]['dominant_gender']
            }
        except Exception as e:
            return {'error': str(e)}
```

### Integracja z ROS2

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class EmotionRecognitionNode(Node):
    def __init__(self):
        super().__init__('emotion_recognition')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(
            String,
            '/perception/emotion',
            10
        )
        self.bridge = CvBridge()
        self.perceiver = AffectivePerception()
    
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        result = self.perceiver.analyze_frame(frame)
        
        emotion_msg = String()
        emotion_msg.data = result.get('emotion', 'unknown')
        self.publisher.publish(emotion_msg)
```

## Wyzwania

### 1. Wieloznaczność Ekspresji
Ta sama ekspresja może oznaczać różne emocje w zależności od kontekstu.

### 2. Różnice Kulturowe
Ekspresja emocji może się różnić między kulturami.

### 3. Mikro-ekspresje
Bardzo krótkie (40-500ms) ekspresje trudne do wykrycia.

### 4. Privacy i Etyka
- Zgoda użytkownika na monitorowanie emocji
- Przechowywanie danych biometrycznych
- Wykorzystanie informacji o emocjach

## Zastosowania

### W Rehabilitacji
- Monitoring stanu emocjonalnego pacjenta
- Wykrywanie frustracji lub zmęczenia
- Dostosowanie intensywności ćwiczeń

### W Opiece nad Seniorami
- Wykrywanie depresji lub anxiety
- Ocena samopoczucia
- Wsparcie emocjonalne

### W Edukacji
- Adaptacyjne systemy e-learning
- Wykrywanie zaangażowania ucznia
- Dostosowanie tempa nauczania

## Narzędzia i Biblioteki

| Narzędzie | Zastosowanie | Link |
|-----------|--------------|------|
| **DeepFace** | Analiza twarzy | [GitHub](https://github.com/serengil/deepface) |
| **MediaPipe** | Face mesh, pose | [Google](https://mediapipe.dev/) |
| **OpenFace** | Facial landmarks | [GitHub](https://github.com/TadasBaltrusaitis/OpenFace) |
| **Py-feat** | Emotional AI | [py-feat.org](https://py-feat.org/) |
| **Affectiva** | Emotion SDK | [Affectiva.com](https://www.affectiva.com/) |

## Etyka i Regulacje

### RODO/GDPR
Rozpoznawanie emocji może być uznane za przetwarzanie danych biometrycznych:
- Wymagana wyraźna zgoda
- Prawo do zapomnienia
- Minimalizacja danych

### IEEE Standards
- P7000 - Model Process for Addressing Ethical Concerns
- P7001 - Transparency of Autonomous Systems

## Kolejne Kroki

- [Rozpoznawanie Emocji](#wiki-emotion-recognition) - szczegóły techniczne
- [Detekcja Twarzy](#wiki-face-detection) - algorytmy wykrywania
- [DeepFace](#wiki-deepface) - biblioteka do analizy twarzy

## Bibliografia

1. Picard, R. W. (1997). *Affective Computing*. MIT Press.
2. Ekman, P. (1992). "An argument for basic emotions". *Cognition & Emotion*, 6(3-4), 169-200.
3. Russell, J. A. (1980). "A circumplex model of affect". *Journal of Personality and Social Psychology*, 39(6), 1161.

---

*Ostatnia aktualizacja: 2025-02-10*  
*Autor: Zespół Percepcji, Laboratorium Robotów Humanoidalnych*
