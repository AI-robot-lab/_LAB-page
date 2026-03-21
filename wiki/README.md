# WIKI - Baza Wiedzy Laboratorium Robotów Humanoidalnych

## Struktura

Wszystkie artykuły są w formacie Markdown (.md) i zorganizowane w następujące kategorie:

### 📁 Robotyka (7 artykułów)
- `ros2.md` - Robot Operating System 2
- `isaac-lab.md` - NVIDIA Isaac Lab symulator
- `unitree-g1.md` - Specyfikacja robota
- `pca-framework.md` - Framework Perception-Cognition-Action
- `slam.md` - Simultaneous Localization and Mapping
- `imu.md` - Inertial Measurement Unit
- `sensor-fusion.md` - Fuzja danych sensorycznych

### 👁️ Percepcja (9 artykułów)
- `computer-vision.md` - Wizja komputerowa
- `affective-computing.md` - Informatyka afektywna
- `emotion-recognition.md` - Rozpoznawanie emocji
- `face-detection.md` - Detekcja twarzy
- `object-detection.md` - Detekcja obiektów
- `pose-estimation.md` - Estymacja pozy
- `pnp.md` - Perspective-n-Point
- `p3p.md` - Perspective-3-Point
- `lidar.md` - LiDAR 3D

### 🧠 Kognicja (7 artykułów)
- `llm.md` - Large Language Models
- `vlm.md` - Vision-Language Models
- `deep-learning.md` - Deep Learning
- `neural-networks.md` - Sieci neuronowe
- `transformers.md` - Architektury Transformer
- `reinforcement-learning.md` - Uczenie przez wzmacnianie
- `transfer-learning.md` - Transfer learning

### ✋ Akcja (6 artykułów)
- `motion-planning.md` - Planowanie ruchu
- `manipulation.md` - Manipulacja robotyczna
- `sim-to-real.md` - Transfer Sim-to-Real
- `control-theory.md` - Teoria sterowania
- `kinematics.md` - Kinematyka robotów
- `trajectory-optimization.md` - Optymalizacja trajektorii

### 💻 Technologie (6 artykułów)
- `pytorch.md` - PyTorch framework
- `opencv.md` - OpenCV biblioteka
- `mediapipe.md` - MediaPipe Google
- `deepface.md` - DeepFace analiza twarzy
- `moveit2.md` - MoveIt2 motion planning
- `docker.md` - Docker dla robotyki

### 👥 Praca w zespole (7 artykułów)
- `praca-w-zespole.md` - Narzędzia, komunikacja i współpraca
- `bash-ssh-jetson.md` - Bash i dostęp do robota po SSH
- `zakladanie-repozytorium-zadaniowego.md` - Tworzenie nowego repozytorium
- `github-releases.md` - GitHub Releases — czym są i jak je tworzyć
- `git-github.md` - Git i GitHub
- `github-tests.md` - Testy w GitHub
- `github_releases_packages.md` - GitHub Releases i Packages

### 🤝 Inne (4 artykuły)
- `hri.md` - Human-Robot Interaction
- `hitl.md` - Hardware-in-the-Loop i Human-in-the-Loop
- `safety.md` - Bezpieczeństwo robotów
- `ethics.md` - Etyka w robotyce

## Format Artykułów

Każdy artykuł markdown zawiera:

```markdown
# Tytuł

## Wprowadzenie
- Definicja
- Zastosowanie w laboratorium

## Główne Sekcje
- Teoria
- Przykłady kodu
- Praktyczne zastosowania

## Powiązane Artykuły
[Link](#wiki-article-id)

## Zasoby
- Linki do dokumentacji
- Tutoriale
- GitHub repos

---
*Ostatnia aktualizacja: YYYY-MM-DD*
*Autor: Zespół XYZ*
```

## Dodawanie Nowych Artykułów

1. **Utwórz plik .md** w katalogu `wiki/`:
   ```bash
   touch wiki/new-article.md
   ```

2. **Dodaj do ARTICLES w wiki.js**:
   ```javascript
   'new-article': 'wiki/new-article.md'
   ```

3. **Dodaj do METADATA w wiki.js**:
   ```javascript
   'new-article': { 
       category: 'Kategoria', 
       title: 'Tytuł Artykułu' 
   }
   ```

4. **Dodaj link w wiki.html**:
   ```html
   <li><a href="#" data-article="new-article">Tytuł Artykułu</a></li>
   ```

## Status Artykułów

✅ **Kompletne** (12):
- ROS2
- Isaac Lab
- PCA Framework
- Computer Vision
- Affective Computing
- Emotion Recognition
- Face Detection
- LiDAR 3D
- LLM
- DeepFace
- PyTorch
- OpenCV

🚧 **W trakcie** (23):
- Pozostałe artykuły wymienione w strukturze

## Wytyczne Pisania

### Styl Kodu
- Python 3.8+
- Type hints gdzie możliwe
- Komentarze po polsku
- Przykłady działające "out of the box"

### Struktura
- H1 tylko tytuł
- H2 dla głównych sekcji
- H3 dla podsekcji
- Code blocks z syntax highlighting

### Linki Wewnętrzne
```markdown
[Nazwa Artykułu](#wiki-article-id)
```

### Obrazy (opcjonalnie)
```markdown
![Alt text](../images/article-name/image.png)
```

## Aktualizacje

Każdy artykuł powinien zawierać datę ostatniej aktualizacji i autora na dole.

---

*Struktura WIKI - Laboratorium Robotów Humanoidalnych PRz*
