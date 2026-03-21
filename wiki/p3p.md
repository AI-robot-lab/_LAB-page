# P3P — Perspective-3-Point

## Wprowadzenie

**P3P** (*Perspective-3-Point*) to specjalny przypadek problemu PnP, w którym
chcemy obliczyć położenie i orientację kamery na podstawie dokładnie
**trzech punktów 3D** oraz ich odpowiadających obserwacji na obrazie 2D.

To jedna z klasycznych metod geometrii widzenia komputerowego. Jest ważna,
bo pokazuje minimalny zestaw informacji potrzebny do oszacowania pozy kamery.

Najprościej można to zapamiętać tak:

- **PnP** = „mam kilka punktów 3D i ich obrazy 2D, chcę znaleźć pozę”,
- **P3P** = „robię to samo, ale używam dokładnie trzech punktów”.

## Intuicja

Załóżmy, że znamy trzy punkty w przestrzeni, na przykład trzy narożniki
obiektu lub trzy charakterystyczne znaczniki. Kamera widzi te punkty na obrazie,
ale sam obraz nie mówi wprost, jak daleko kamera jest od każdego z nich.

P3P korzysta z faktu, że:

- znamy odległości między punktami 3D,
- znamy kierunki promieni wychodzących z kamery do punktów widocznych na obrazie,
- z tych zależności można odtworzyć możliwe położenia kamery.

To trochę jak triangulacja „odwrotna”: zamiast wyznaczać punkt z wielu kamer,
wyznaczamy położenie jednej kamery na podstawie znanych punktów sceny.

## Dlaczego akurat trzy punkty

Trzy punkty to przypadek minimalny. Mniejsza liczba punktów nie wystarcza do
jednoznacznego określenia pełnej pozy 6D kamery.

Jednocześnie trzeba pamiętać, że „minimalny” nie znaczy „najlepszy”:

- metoda jest szybka,
- dobrze nadaje się do algorytmów RANSAC,
- ale bywa bardziej wrażliwa na szum niż rozwiązania korzystające z większej
  liczby punktów.

W praktyce często używa się P3P jako kroku wstępnego, a potem wynik doprecyzowuje
się metodą iteracyjną lub dodatkowymi punktami.

## Jakie dane są potrzebne

### Punkty 3D

Trzy punkty muszą mieć znane współrzędne w układzie obiektu:

```text
A = (x1, y1, z1)
B = (x2, y2, z2)
C = (x3, y3, z3)
```

### Punkty 2D na obrazie

Potrzebujemy odpowiadających im obserwacji na zdjęciu lub klatce wideo:

```text
a = (u1, v1)
b = (u2, v2)
c = (u3, v3)
```

### Skalibrowana kamera

P3P zakłada znajomość parametrów kamery, aby z pikseli wyznaczyć kierunki
promieni obserwacji.

## Co zwraca P3P

To bardzo ważna cecha: **P3P może zwracać kilka możliwych rozwiązań**.

Dlaczego? Bo z samej geometrii trzech punktów czasem istnieje więcej niż jedno
ustawienie kamery zgodne z obserwacją. Dlatego po obliczeniu kandydatów trzeba
wybrać rozwiązanie poprawne, na przykład:

- używając czwartego punktu kontrolnego,
- sprawdzając błąd reprojekcji,
- korzystając z wiedzy o scenie,
- doprecyzowując wynik metodą iteracyjną.

To odróżnia P3P od potocznego wyobrażenia, że „algorytm zawsze daje jedną
jedyną odpowiedź”. W minimalnym problemie geometria bywa niejednoznaczna.

## Gdzie P3P jest przydatne

### W RANSAC

W algorytmach odpornych na błędne dopasowania często losuje się małe podzbiory
punktów i dla nich liczy kandydacką pozę. P3P jest tu bardzo wygodne, bo działa
na minimalnej liczbie punktów i jest szybkie.

### W szybkiej inicjalizacji pozy

Gdy system musi błyskawicznie uzyskać przybliżoną pozę kamery, P3P może dostarczyć
pierwszego oszacowania, które potem zostanie poprawione.

### W systemach markerowych i SLAM

Jeśli wykryto kilka pewnych punktów charakterystycznych, P3P może posłużyć do
obliczenia kandydatów pozy kamery względem mapy lub markera.

## Ograniczenia P3P

### Wrażliwość na szum

Ponieważ korzystamy tylko z trzech punktów, każdy błąd detekcji ma duży wpływ na
wynik końcowy.

### Możliwość wielu rozwiązań

Sam wynik P3P często wymaga dodatkowej weryfikacji.

### Słaba geometria punktów

Jeśli punkty są źle rozmieszczone, bardzo blisko siebie albo prawie współliniowe,
rozwiązanie może być niestabilne lub niepraktyczne.

## P3P a klasyczne PnP

Poniższa tabela dobrze pokazuje różnicę.

| Cecha | P3P | Ogólne PnP |
|---|---|---|
| Liczba punktów | Dokładnie 3 punkty (czasem 4. punkt do wyboru rozwiązania) | 4 lub więcej, zależnie od metody |
| Szybkość | Bardzo wysoka | Zależy od wariantu |
| Odporność na szum | Mniejsza | Zwykle większa przy większej liczbie punktów |
| Liczba rozwiązań | Może być wiele kandydatów | Często jedno rozwiązanie po optymalizacji |
| Typowe użycie | RANSAC, inicjalizacja | Finalna estymacja pozy |

## Przykład z OpenCV

W OpenCV można użyć P3P jako jednego z trybów `solvePnP`.

```python
import cv2
import numpy as np

object_points = np.array([
    [0.0, 0.0, 0.0],
    [0.2, 0.0, 0.0],
    [0.0, 0.2, 0.0],
    [0.2, 0.2, 0.0]
], dtype=np.float32)

image_points = np.array([
    [312.0, 244.0],
    [455.0, 251.0],
    [298.0, 390.0],
    [446.0, 402.0]
], dtype=np.float32)

camera_matrix = np.array([
    [850.0,   0.0, 320.0],
    [  0.0, 850.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float32)

dist_coeffs = np.zeros((4, 1), dtype=np.float32)

success, rvec, tvec = cv2.solvePnP(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs,
    flags=cv2.SOLVEPNP_P3P
)

if success:
    print("Kandydat pozy:")
    print("rvec:", rvec.ravel())
    print("tvec:", tvec.ravel())
```

Choć nazwa metody mówi o trzech punktach, w praktycznych bibliotekach często
podaje się **co najmniej cztery korespondencje**, aby łatwiej odrzucić błędne
rozwiązania i wybrać ten kandydat, który najlepiej zgadza się z obrazem.

## Jak myśleć o P3P w praktyce

Jeśli dopiero zaczynasz pracę z robotyką i wizją komputerową, warto zapamiętać
następujący model mentalny:

1. Kamera widzi punkty na obrazie.
2. Kalibracja mówi, w jakich kierunkach kamera „patrzy” na te punkty.
3. Znana geometria trzech punktów 3D ogranicza możliwe ustawienia kamery.
4. Algorytm wylicza jedną lub kilka możliwych póz.
5. Dodatkowe dane wybierają rozwiązanie właściwe.

## Kiedy lepiej użyć czegoś innego

P3P nie zawsze jest najlepszym wyborem. Jeśli masz więcej punktów i zależy Ci na
stabilności, zwykle lepiej użyć:

- iteracyjnego `solvePnP`,
- `solvePnPRansac`,
- EPnP lub innych wariantów dla większej liczby korespondencji.

P3P jest świetne jako narzędzie pomocnicze, ale nie zawsze jako końcowy etap
estymacji pozy.

## Podsumowanie

P3P to minimalistyczny, szybki i bardzo ważny wariant problemu estymacji pozy
kamery. Używa trzech punktów 3D i ich obrazów 2D, aby wyznaczyć możliwe
ustawienia kamery w przestrzeni. Najczęściej spotyka się go w RANSAC-u,
inicjalizacji pozy i klasycznych zadaniach geometrii widzenia komputerowego.

## Powiązane artykuły

- [PnP — Perspective-n-Point](#wiki-pnp)
- [Estymacja pozy](#wiki-pose-estimation)
- [Markery wizyjne w nawigacji](#wiki-markery_wizyjne_nawigacja)
- [SLAM — Lokalizacja i Mapowanie](#wiki-slam)

## Zasoby

- Dokumentacja OpenCV: tryby `solvePnP`
- Materiały o problemie Perspective-3-Point
- Wprowadzenia do geometrii wielowidokowej i estymacji pozy

---
*Ostatnia aktualizacja: 2026-03-21*
*Autor: OpenAI Codex*
