# HRI - Human-Robot Interaction

## Wprowadzenie

**Human-Robot Interaction (HRI)** bada interakcję między ludźmi a robotami, obejmując komunikację, współpracę i bezpieczeństwo.

## Modalności Interakcji

### Komunikacja Werbalna

```python
import speech_recognition as sr
from gtts import gTTS
import pygame

class VoiceInterface:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        pygame.mixer.init()
    
    def listen(self):
        """
        Listen to user speech
        """
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
        
        try:
            text = self.recognizer.recognize_google(audio, language='pl-PL')
            return text
        except:
            return None
    
    def speak(self, text):
        """
        Text-to-speech
        """
        tts = gTTS(text=text, lang='pl')
        tts.save('response.mp3')
        
        pygame.mixer.music.load('response.mp3')
        pygame.mixer.music.play()
```

### Rozpoznawanie Gestów

```python
import mediapipe as mp

class GestureRecognition:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
    
    def recognize_gesture(self, image):
        """
        Recognize hand gesture
        """
        results = self.hands.process(image)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            
            # Simple gesture: pointing
            index_tip = landmarks.landmark[8]
            index_base = landmarks.landmark[5]
            
            if index_tip.y < index_base.y:
                return "POINTING_UP"
            
            return "OTHER"
        
        return "NO_HAND"
```

### Detekcja Intencji

```python
from transformers import pipeline

class IntentDetector:
    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
    
    def detect_intent(self, text):
        """
        Classify user intent
        """
        candidate_labels = [
            "greeting",
            "command",
            "question",
            "thanks",
            "goodbye"
        ]
        
        result = self.classifier(text, candidate_labels)
        
        return result['labels'][0]
```

## Bezpieczeństwo w HRI

### Detekcja Człowieka

```python
class SafetyMonitor:
    def __init__(self, safety_distance=1.0):
        self.safety_distance = safety_distance
        self.detector = YOLOv8('yolov8n.pt')
    
    def check_human_proximity(self, image, robot_position):
        """
        Check if human is too close
        """
        detections = self.detector(image)
        
        for det in detections:
            if det['class'] == 'person':
                # Estimate distance (simple)
                bbox_height = det['bbox'][3] - det['bbox'][1]
                distance = 1000 / bbox_height  # Rough estimate
                
                if distance < self.safety_distance:
                    return True, "EMERGENCY_STOP"
        
        return False, "SAFE"
```

### Collaborative Control

```python
class CollaborativeControl:
    def __init__(self):
        self.robot_velocity = 0.0
        self.max_velocity = 1.0
    
    def adjust_velocity(self, human_distance):
        """
        Adjust robot velocity based on human proximity
        """
        if human_distance < 0.5:
            self.robot_velocity = 0.0
        elif human_distance < 1.0:
            self.robot_velocity = 0.3 * self.max_velocity
        elif human_distance < 2.0:
            self.robot_velocity = 0.6 * self.max_velocity
        else:
            self.robot_velocity = self.max_velocity
        
        return self.robot_velocity
```

## Social Navigation

```python
class SocialNavigator:
    def __init__(self):
        self.personal_space = 1.2  # meters
        self.social_space = 3.5
    
    def plan_socially_aware_path(self, robot_pos, goal, humans):
        """
        Plan path considering social norms
        """
        # Add virtual obstacles around humans
        virtual_obstacles = []
        
        for human in humans:
            # Personal space
            virtual_obstacles.append({
                'center': human['position'],
                'radius': self.personal_space
            })
        
        # Plan with RRT avoiding virtual obstacles
        path = self.rrt_plan(robot_pos, goal, virtual_obstacles)
        
        return path
```

## Powiązane Artykuły

- [Affective Computing](#wiki-affective-computing)
- [Emotion Recognition](#wiki-emotion-recognition)
- [Safety](#wiki-safety)

---

*Ostatnia aktualizacja: 2025-02-11*
