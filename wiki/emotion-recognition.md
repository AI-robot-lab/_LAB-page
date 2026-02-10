# Rozpoznawanie emocji

## Wprowadzenie

**Rozpoznawanie emocji** (Emotion Recognition) to proces automatycznej analizy i klasyfikacji stanów emocjonalnych człowieka na podstawie różnych modalności: twarzy, głosu, tekstu czy sygnałów fizjologicznych.

## Metody Rozpoznawania

### 1. Analiza ekspresji twarzy (FER)

#### Facial Action Coding System (FACS)

FACS dzieli ruchy twarzy na **Action Units (AU)** - podstawowe jednostki ruchu mięśni:

| AU | Nazwa | Emocja |
|----|-------|---------|
| AU1 | Podniesienie wewnętrznej brwi | Smutek, strach |
| AU4 | Obniżenie brwi | Gniew |
| AU6 | Uniesienie policzków | Radość |
| AU12 | Rozciągnięcie kącików ust | Radość |
| AU15 | Obniżenie kącików ust | Smutek |

#### Deep Learning dla FER

```python
import cv2
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1

class EmotionRecognizer:
    def __init__(self, model_path='models/emotion_model.pth'):
        # Detektor twarzy
        self.mtcnn = MTCNN(keep_all=True, device='cuda')
        
        # Model rozpoznawania emocji
        self.model = torch.load(model_path)
        self.model.eval()
        
        # Klasy emocji
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 
                         'sad', 'surprise', 'neutral']
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def recognize(self, image):
        # Detekcja twarzy
        boxes, probs = self.mtcnn.detect(image)
        
        if boxes is None:
            return None
        
        results = []
        for box in boxes:
            # Wycinanie twarzy
            x1, y1, x2, y2 = [int(b) for b in box]
            face = image[y1:y2, x1:x2]
            
            # Preprocessing
            face_tensor = self.transform(face).unsqueeze(0)
            
            # Predykcja
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                emotion_idx = torch.argmax(probs).item()
                confidence = probs[0][emotion_idx].item()
            
            results.append({
                'bbox': box,
                'emotion': self.emotions[emotion_idx],
                'confidence': confidence,
                'probabilities': {
                    emotion: prob.item() 
                    for emotion, prob in zip(self.emotions, probs[0])
                }
            })
        
        return results

# Użycie
recognizer = EmotionRecognizer()
image = cv2.imread('face.jpg')
emotions = recognizer.recognize(image)

for emotion_data in emotions:
    print(f"Emocja: {emotion_data['emotion']}")
    print(f"Pewność: {emotion_data['confidence']:.2f}")
```

### 2. Rozpoznawanie Emocji z Głosu

#### Cechy Akustyczne

```python
import librosa
import numpy as np

def extract_voice_features(audio_path):
    # Wczytaj audio
    y, sr = librosa.load(audio_path, sr=22050)
    
    # MFCC - Mel-frequency cepstral coefficients
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    # Pitch (F0)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0])
    
    # Energy
    energy = np.sum(y ** 2) / len(y)
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    
    # Spektralna centroida
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroids)
    
    return {
        'mfcc_mean': mfccs_mean,
        'mfcc_std': mfccs_std,
        'pitch_mean': pitch_mean,
        'energy': energy,
        'zcr_mean': zcr_mean,
        'spectral_centroid': spectral_centroid_mean
    }

# Klasyfikacja
from sklearn.ensemble import RandomForestClassifier

# Trenowanie modelu (przykład)
features_train = [...]  # ekstraktowane cechy
labels_train = [...]    # emocje ('angry', 'happy', etc.)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(features_train, labels_train)

# Predykcja
features_test = extract_voice_features('test_audio.wav')
emotion = clf.predict([list(features_test.values())])
```

### 3. Rozpoznawanie Emocji z Tekstu

#### NLP dla Sentiment Analysis

```python
from transformers import pipeline

# Model pre-trenowany do analizy sentymentu
classifier = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-emotion",
    device=0  # GPU
)

text = "Jestem bardzo szczęśliwy dzisiaj!"
result = classifier(text)

print(f"Emocja: {result[0]['label']}")
print(f"Pewność: {result[0]['score']:.2f}")
```

#### BERT dla Emocji

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class TextEmotionRecognizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "j-hartmann/emotion-english-distilroberta-base"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "j-hartmann/emotion-english-distilroberta-base"
        )
        self.emotions = ['anger', 'disgust', 'fear', 'joy', 
                        'neutral', 'sadness', 'surprise']
    
    def predict(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        emotion_idx = torch.argmax(probs).item()
        
        return {
            'emotion': self.emotions[emotion_idx],
            'confidence': probs[0][emotion_idx].item(),
            'all_scores': {
                emotion: score.item() 
                for emotion, score in zip(self.emotions, probs[0])
            }
        }
```

## Multimodalne Rozpoznawanie Emocji

### Fuzja Modalności

```python
class MultimodalEmotionRecognizer:
    def __init__(self):
        self.face_recognizer = FaceEmotionRecognizer()
        self.voice_recognizer = VoiceEmotionRecognizer()
        self.text_recognizer = TextEmotionRecognizer()
        
        # Wagi dla poszczególnych modalności
        self.weights = {
            'face': 0.4,
            'voice': 0.3,
            'text': 0.3
        }
    
    def fuse_predictions(self, face_probs, voice_probs, text_probs):
        # Ważona średnia rozkładów prawdopodobieństw
        fused_probs = (
            self.weights['face'] * face_probs +
            self.weights['voice'] * voice_probs +
            self.weights['text'] * text_probs
        )
        
        emotion_idx = np.argmax(fused_probs)
        return self.emotions[emotion_idx], fused_probs[emotion_idx]
    
    def recognize(self, image, audio, text):
        # Predykcje z różnych modalności
        face_emotion = self.face_recognizer.predict(image)
        voice_emotion = self.voice_recognizer.predict(audio)
        text_emotion = self.text_recognizer.predict(text)
        
        # Fuzja
        final_emotion, confidence = self.fuse_predictions(
            face_emotion['probabilities'],
            voice_emotion['probabilities'],
            text_emotion['probabilities']
        )
        
        return {
            'emotion': final_emotion,
            'confidence': confidence,
            'modalities': {
                'face': face_emotion,
                'voice': voice_emotion,
                'text': text_emotion
            }
        }
```

## Bazy Danych

### Popularne Datasety

| Dataset | Modalność | Liczba próbek | Emocje |
|---------|-----------|---------------|--------|
| **FER2013** | Twarz | 35,887 | 7 emocji |
| **AffectNet** | Twarz | 1M+ | 8 emocji + VA |
| **RAVDESS** | Audio-Wizual | 7,356 | 8 emocji |
| **IEMOCAP** | Multimodalny | 12 godz. | 10 emocji |
| **EmoReact** | Wideo | 1,102 | 25 emocji |

### Download FER2013

```bash
# Kaggle API
pip install kaggle

# Download
kaggle datasets download -d msambare/fer2013

# Unzip
unzip fer2013.zip -d data/fer2013/
```

## Wyzwania

### 1. Niska Jakość Danych
- Złe oświetlenie
- Okludowanie twarzy
- Szum w audio

### 2. Zmienność Między Osobami
- Różnice kulturowe
- Indywidualne style ekspresji

### 3. Mikro-Ekspresje
- Bardzo krótkie (40-500ms)
- Trudne do uchwycenia

### 4. Kontekst
- Ta sama ekspresja = różne emocje w różnych kontekstach

## Zastosowania w Robotyce

```python
# Integracja z systemem robotycznym
class EmotionAwareRobot:
    def __init__(self):
        self.emotion_recognizer = MultimodalEmotionRecognizer()
        self.behavior_controller = RobotBehaviorController()
    
    def interact(self, user_input):
        # Rozpoznaj emocję
        emotion_data = self.emotion_recognizer.recognize(
            image=user_input['camera'],
            audio=user_input['microphone'],
            text=user_input['speech_to_text']
        )
        
        current_emotion = emotion_data['emotion']
        
        # Dostosuj zachowanie robota
        if current_emotion in ['sad', 'fear']:
            self.behavior_controller.set_empathetic_mode()
            response = "Rozumiem, że możesz się źle czuć. Jak mogę pomóc?"
        
        elif current_emotion == 'angry':
            self.behavior_controller.set_calming_mode()
            response = "Widzę, że jesteś zdenerwowany. Spróbujmy to omówić spokojnie."
        
        elif current_emotion == 'happy':
            self.behavior_controller.set_enthusiastic_mode()
            response = "Cieszę się, że jesteś w dobrym humorze!"
        
        else:
            self.behavior_controller.set_neutral_mode()
            response = "Jak mogę Ci dzisiaj pomóc?"
        
        return response
```

## Metryki Ewaluacji

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=emotions)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=emotions,
    yticklabels=emotions
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Emotion Recognition Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred, target_names=emotions))
```

## Powiązane Artykuły

- [Informatyka Afektywna](#wiki-affective-computing) - podstawy teoretyczne
- [Detekcja Twarzy](#wiki-face-detection) - preprocessing dla FER
- [DeepFace](#wiki-deepface) - biblioteka do analizy twarzy

## Bibliografia

1. Ekman, P., & Friesen, W. V. (1978). *Facial Action Coding System*.
2. Mollahosseini, A., Hasani, B., & Mahoor, M. H. (2017). "AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild". *IEEE TAFFC*.

---

*Ostatnia aktualizacja: 2025-02-10*  
*Autor: Zespół Percepcji, Laboratorium Robotów Humanoidalnych*
