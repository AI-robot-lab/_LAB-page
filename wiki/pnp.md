# PnP — Perspective-n-Point

## Wprowadzenie

**PnP** (*Perspective-n-Point*) to rodzina metod, które pozwalają obliczyć
**położenie** i **orientację** kamery względem obiektu, jeśli znamy:

1. pozycje kilku punktów obiektu w przestrzeni 3D,
2. miejsca tych samych punktów na obrazie 2D,
3. parametry kamery po kalibracji.

W praktyce PnP odpowiada na proste pytanie:

> „Skoro wiem, gdzie narożniki obiektu są w rzeczywistości i widzę je na
> zdjęciu, to gdzie znajduje się kamera?”.

W robotyce humanoidalnej jest to bardzo ważne, ponieważ robot dzięki PnP może:

- określić odległość do markera lub narzędzia,
- wyznaczyć orientację obiektu przed chwytem,
- oszacować pozycję głowy, dłoni lub kamery,
- połączyć obserwacje z kamery z mapą otoczenia.

## Intuicja bez skomplikowanej matematyki

Wyobraź sobie kartkę z narysowanym kwadratem. Znasz dokładnie położenie
czterech narożników tego kwadratu w układzie 3D obiektu. Gdy zrobisz zdjęcie,
kwadrat na obrazie będzie zwykle zdeformowany przez perspektywę:

- gdy kamera patrzy na wprost — kształt przypomina regularny kwadrat,
- gdy kamera patrzy pod kątem — kwadrat zamienia się w trapez,
- gdy kamera jest bliżej — obiekt zajmuje więcej miejsca na obrazie.

Algorytm PnP wykorzystuje właśnie tę deformację perspektywiczną, aby odtworzyć,
jak kamera musiała być ustawiona względem obiektu.

## Jakie dane wejściowe są potrzebne

Aby uruchomić PnP, potrzebujemy trzech grup danych.

### 1. Punkty 3D obiektu

To współrzędne punktów w lokalnym układzie odniesienia obiektu, np. narożniki
markera AprilTag, planszy, kostki albo modelu twarzy.

```text
P1 = (0, 0, 0)
P2 = (0.1, 0, 0)
P3 = (0.1, 0.1, 0)
P4 = (0, 0.1, 0)
```

### 2. Odpowiadające im punkty 2D na obrazie

To pozycje tych samych punktów wykrytych na klatce z kamery, np. w pikselach:

```text
p1 = (412, 215)
p2 = (533, 226)
p3 = (520, 351)
p4 = (398, 338)
```

### 3. Parametry wewnętrzne kamery

Są to dane pochodzące z kalibracji kamery:

- ogniskowa w pikselach,
- współrzędne środka obrazu,
- współczynniki dystorsji.

Bez kalibracji PnP zwykle działa zauważalnie gorzej, bo algorytm nie wie,
jak kamera zniekształca obraz.

## Co jest wynikiem PnP

Typowy solver PnP zwraca dwa elementy:

- **wektor rotacji** (`rvec`) — mówi, jak obrócony jest obiekt lub kamera,
- **wektor translacji** (`tvec`) — mówi, gdzie obiekt znajduje się względem kamery.

Najczęściej interpretujemy to tak:

- `tvec[0]` — przesunięcie w lewo/prawo,
- `tvec[1]` — przesunięcie góra/dół,
- `tvec[2]` — odległość od kamery.

Jeśli `tvec[2]` maleje, obiekt jest bliżej. Jeśli `rvec` wskazuje duży obrót,
znaczy to, że obiekt jest obserwowany pod kątem.

## Dlaczego potrzeba kilku punktów

Jeden punkt 3D widoczny na obrazie nie wystarcza — przez pojedynczy piksel można
poprowadzić wiele możliwych promieni obserwacji. Dopiero kilka odpowiadających
sobie punktów pozwala jednoznaczniej wyznaczyć pozę.

W praktyce:

- **3 punkty** to przypadek minimalny dla niektórych metod,
- **4 punkty** są bardzo popularne przy markerach kwadratowych,
- **więcej niż 4 punkty** zwykle daje większą odporność na szum i błędy detekcji.

## Gdzie PnP jest używane w laboratorium

### Markery wizyjne

Jeśli robot widzi marker o znanym rozmiarze, można użyć narożników markera jako
par punktów 3D–2D i policzyć jego położenie względem kamery.

### Chwytanie obiektów

Jeśli znamy model 3D obiektu i wykryjemy charakterystyczne punkty na obrazie,
PnP pomaga ustawić chwytak pod odpowiednim kątem.

### Analiza twarzy i głowy

W systemach śledzenia twarzy można użyć punktów charakterystycznych, takich jak
kąciki oczu czy czubek nosa, aby oszacować kierunek patrzenia.

### Łączenie wizji z mapą

Pozycja markera albo obiektu z PnP może być przeliczana do globalnego układu
robota i łączona z SLAM-em lub odometrią.

## Najprostszy przykład z OpenCV

```python
import cv2
import numpy as np

object_points = np.array([
    [0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.1, 0.1, 0.0],
    [0.0, 0.1, 0.0]
], dtype=np.float32)

image_points = np.array([
    [412.0, 215.0],
    [533.0, 226.0],
    [520.0, 351.0],
    [398.0, 338.0]
], dtype=np.float32)

camera_matrix = np.array([
    [900.0,   0.0, 320.0],
    [  0.0, 900.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float32)

dist_coeffs = np.zeros((4, 1), dtype=np.float32)

success, rvec, tvec = cv2.solvePnP(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)

if success:
    print("Rotacja:", rvec.ravel())
    print("Translacja:", tvec.ravel())
```

W powyższym przykładzie OpenCV szuka takiej pozy kamery, dla której punkty 3D
po rzutowaniu na obraz jak najlepiej pokryją się z wykrytymi punktami 2D.

## Najczęstsze warianty metod PnP

W OpenCV i literaturze można spotkać kilka odmian:

- **Iterative PnP** — dokładna metoda optymalizacyjna, często używana domyślnie,
- **EPnP** — szybka metoda dla większej liczby punktów,
- **P3P** — wariant minimalny używający trzech punktów,
- **AP3P** — alternatywna wersja P3P,
- **RANSAC + PnP** — odporność na błędne dopasowania punktów.

W praktyce wiele systemów używa `solvePnPRansac`, bo pojedynczy źle wykryty punkt
może mocno zepsuć wynik zwykłego dopasowania.

## Typowe problemy i błędy

### Zła kalibracja kamery

Jeśli macierz kamery lub dystorsja są błędne, wynik pozycji będzie niestabilny
albo przesunięty względem rzeczywistości.

### Źle dopasowane punkty 2D–3D

Kolejność punktów musi być zgodna. Jeśli narożniki markera zostaną podane w złej
kolejności, algorytm obliczy niepoprawną pozę.

### Punkty leżące prawie w jednej linii

Gdy punkty mają zbyt mało informacji geometrycznej, rozwiązanie staje się
niestabilne. Dlatego dobrze wybierać punkty rozłożone przestrzennie.

### Szum i rozmycie obrazu

Niewyraźne narożniki albo słaba detekcja obniżają dokładność końcowej estymacji.

## Jak rozumieć relację między PnP a P3P

- **PnP** to szeroka klasa problemów i metod.
- **P3P** to szczególny przypadek PnP, w którym używa się minimalnej liczby
  punktów potrzebnych do wyznaczenia pozy.

Można więc powiedzieć, że:

> Każde P3P jest problemem typu PnP, ale nie każde PnP jest P3P.

## Podsumowanie

PnP to podstawowe narzędzie geometrii widzenia komputerowego. Pozwala przejść od
prostego obrazu 2D do praktycznej informacji 3D: gdzie znajduje się obiekt i jak
jest obrócony względem kamery. W robotyce to jedna z najważniejszych metod do
lokalizacji markerów, estymacji pozy obiektów i integracji percepcji z ruchem.

## Powiązane artykuły

- [P3P — Perspective-3-Point](#wiki-p3p)
- [Estymacja pozy](#wiki-pose-estimation)
- [Markery wizyjne w nawigacji](#wiki-markery_wizyjne_nawigacja)
- [Kalibracja kamery robota](#wiki-kalibracja_kamery)
- [SLAM — Lokalizacja i Mapowanie](#wiki-slam)

## Zasoby

- Dokumentacja OpenCV: `solvePnP`, `solvePnPRansac`
- Materiały o geometrii kamery i projekcji perspektywicznej
- Tutoriale dotyczące markerów fiducjalnych i estymacji pozy

---
*Ostatnia aktualizacja: 2026-03-21*
*Autor: OpenAI Codex*
