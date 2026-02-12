# 📸 Robot Images - Instructions

## Zdjęcia do Pobrania

Pobierz poniższe zdjęcia Unitree G1 z serwera PRz i umieść je w tym folderze:

### 1. robot-1.jpg
**URL:** https://prz.edu.pl/thumb/ODWWMBLDAKdgoBVwZmAlkDWg1XEjMg,1/pl/news/826/77/1/LDVQNIxwIeQdlSEVqWFM,ly5a1922.jpg

**Opis:** Unitree G1 - widok z przodu podczas demonstracji w laboratorium

**Użycie:**
- Hero section na stronie głównej
- Galeria robota (pierwsze zdjęcie)

---

### 2. robot-2.jpg
**URL:** https://prz.edu.pl/thumb/QISnASPyMZZRkSRBV1EUoQSRREASAz,1/pl/news/826/77/1/LDVQNIxwIeQdlSEVqWFM,ly5a1928.jpg

**Opis:** Unitree G1 - widok boczny z systemem sensorycznym

**Użycie:**
- Galeria robota (drugie zdjęcie)
- Ilustracja systemu percepcji (LiDAR, kamery RGB-D)

---

### 3. robot-3.jpg
**URL:** https://prz.edu.pl/thumb/MqblQ2Gwc9QT02YDFRNW41ZjBgJQQX,1/pl/news/826/77/1/LDVQNIxwIeQdlSEVqWFM,ly5a1898.jpg

**Opis:** Unitree G1 - prezentacja manipulacji z dłońmi Dex3-1

**Użycie:**
- Galeria robota (trzecie zdjęcie)
- Ilustracja precyzyjnej manipulacji

---

### wget (Linux/Mac)
```bash
cd assets/images/robot/

wget -O robot-1.jpg "https://prz.edu.pl/thumb/ODWWMBLDAKdgoBVwZmAlkDWg1XEjMg,1/pl/news/826/77/1/LDVQNIxwIeQdlSEVqWFM,ly5a1922.jpg"

wget -O robot-2.jpg "https://prz.edu.pl/thumb/QISnASPyMZZRkSRBV1EUoQSRREASAz,1/pl/news/826/77/1/LDVQNIxwIeQdlSEVqWFM,ly5a1928.jpg"

wget -O robot-3.jpg "https://prz.edu.pl/thumb/MqblQ2Gwc9QT02YDFRNW41ZjBgJQQX,1/pl/news/826/77/1/LDVQNIxwIeQdlSEVqWFM,ly5a1898.jpg"
```

### Metoda 3: curl (Windows/Linux/Mac)
```bash
cd assets/images/robot/

curl -L -o robot-1.jpg "https://prz.edu.pl/thumb/ODWWMBLDAKdgoBVwZmAlkDWg1XEjMg,1/pl/news/826/77/1/LDVQNIxwIeQdlSEVqWFM,ly5a1922.jpg"

curl -L -o robot-2.jpg "https://prz.edu.pl/thumb/QISnASPyMZZRkSRBV1EUoQSRREASAz,1/pl/news/826/77/1/LDVQNIxwIeQdlSEVqWFM,ly5a1928.jpg"

curl -L -o robot-3.jpg "https://prz.edu.pl/thumb/MqblQ2Gwc9QT02YDFRNW41ZjBgJQQX,1/pl/news/826/77/1/LDVQNIxwIeQdlSEVqWFM,ly5a1898.jpg"
```

---

## Zalecenia Techniczne

### Format:
- JPG lub PNG
- RGB color space

### Rozmiar:
- Szerokość: 800-1200px (zalecane)
- Wysokość: proporcjonalna
- Maksymalny rozmiar pliku: 500KB każdy

### Optymalizacja:
```bash
# Opcjonalnie: zmniejsz rozmiar bez utraty jakości
convert robot-1.jpg -quality 85 -resize 1200x robot-1.jpg
```

---

## Fallback Images

Strona zawiera fallback images na wypadek błędu ładowania:
- Placeholder images generowane automatycznie
- Graceful degradation - strona działa bez zdjęć

---

## Status: ⚠️ **IMAGES NEEDED**

Po dodaniu zdjęć:
1. Sprawdź czy wyświetlają się poprawnie
2. Zmień status na: ✅ **IMAGES READY**
3. Usuń ten README (opcjonalnie)

---

*Laboratorium Robotów Humanoidalnych PRz*  
*2025-02-12*
