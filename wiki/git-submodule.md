# Git Submodule — zarządzanie zagnieżdżonymi repozytoriami

Git submodule to mechanizm pozwalający umieścić jedno repozytorium Git wewnątrz innego jako zależność. Dzięki temu projekt nadrzędny (*superprojekt*) może precyzyjnie wskazać konkretny commit zależności, zachowując pełną historię każdego z repozytoriów oddzielnie.

---

## Spis treści

1. [Czym jest submodule?](#1-czym-jest-submodule)
2. [Kiedy używać submodule?](#2-kiedy-używać-submodule)
3. [Dodawanie submodule](#3-dodawanie-submodule)
4. [Klonowanie projektu z submodule](#4-klonowanie-projektu-z-submodule)
5. [Aktualizacja submodule](#5-aktualizacja-submodule)
6. [Praca wewnątrz submodule](#6-praca-wewnątrz-submodule)
7. [Usuwanie submodule](#7-usuwanie-submodule)
8. [Plik .gitmodules](#8-plik-gitmodules)
9. [Typowe problemy i rozwiązania](#9-typowe-problemy-i-rozwiązania)
10. [Dobre praktyki](#10-dobre-praktyki)
11. [Ściągawka komend](#11-ściągawka-komend)

---

## 1. Czym jest submodule?

Submodule to **wskaźnik na konkretny commit** w zewnętrznym repozytorium Git. Superprojekt przechowuje jedynie:

- ścieżkę do podkatalogu, w którym ma być umieszczone zewnętrzne repo,
- adres URL zewnętrznego repozytorium,
- SHA commitu, na który submodule ma być „przypięty".

Sam kod submodule nie jest kopiowany do superprojektu — jest tylko *referencja*. Dzięki temu możliwa jest niezależna historia zmian i niezależne wersjonowanie każdego komponentu.

```text
superprojekt/
├── .git/
├── .gitmodules          ← definicje submodule
├── src/
│   └── main.py
└── libs/
    └── shared-utils/    ← katalog submodule (zewnętrzne repo)
        ├── .git/
        └── utils.py
```

**Opis schematu:** Katalog `libs/shared-utils/` wygląda jak zwykły folder, ale zawiera własne repozytorium Git. Superprojekt śledzi tylko, który commit tego repo jest aktualnie używany.

---

## 2. Kiedy używać submodule?

| Sytuacja | Uzasadnienie |
|---|---|
| Wspólna biblioteka używana przez kilka teamów | Każdy projekt wskazuje na wybraną, stabilną wersję biblioteki |
| Zewnętrzne zależności z własnym cyklem życia | Niezależne wersjonowanie i historia |
| Dane treningowe lub modele ML trzymane osobno | Duże pliki nie zaśmiecają głównego repo |
| Wspólny moduł konfiguracji/infrastruktury | Łatwa synchronizacja między projektami |

> **Kiedy NIE używać:** jeśli zależność zmienia się bardzo często razem z projektem głównym, lepszym rozwiązaniem jest zwykły pakiet (np. pip, npm) lub monorepo.

---

## 3. Dodawanie submodule

### Podstawowe dodanie

```bash
git submodule add <URL-repozytorium> <ścieżka>
```

Przykład — dodanie biblioteki narzędziowej do katalogu `libs/shared-utils`:

```bash
git submodule add https://github.com/AI-robot-lab/shared-utils.git libs/shared-utils
```

Po wykonaniu komendy Git:

1. Klonuje wskazane repozytorium do podanej ścieżki.
2. Tworzy (lub uzupełnia) plik `.gitmodules`.
3. Dodaje wpis w `.git/config`.
4. Rejestruje bieżący commit submodule jako nową zmianę do zatwierdzenia.

### Zatwierdzenie zmian

```bash
git add .gitmodules libs/shared-utils
git commit -m "feat: dodanie submodule shared-utils"
```

### Wskazanie konkretnej gałęzi

```bash
git submodule add -b main https://github.com/AI-robot-lab/shared-utils.git libs/shared-utils
```

Opcja `-b` powoduje, że podczas `git submodule update --remote` pobierana jest najnowsza wersja z podanej gałęzi.

---

## 4. Klonowanie projektu z submodule

Zwykłe `git clone` nie pobiera zawartości submodule — katalogi submodule pozostają puste. Są dwa sposoby na poprawne sklonowanie projektu.

### Opcja A — rekurencyjne klonowanie (zalecane)

```bash
git clone --recurse-submodules <URL-superprojektu>
```

**Opis:** Jednoczesne sklonowanie superprojektu i wszystkich zarejestrowanych submodule. Każde submodule jest inicjalizowane i wypełnione od razu.

### Opcja B — inicjalizacja po klonowaniu

```bash
git clone <URL-superprojektu>
cd <katalog>
git submodule init
git submodule update
```

lub krócej:

```bash
git clone <URL-superprojektu>
cd <katalog>
git submodule update --init --recursive
```

**Opis:** Najpierw klonujesz superprojekt, a następnie ręcznie inicjalizujesz i wypełniasz submodule. Opcja `--recursive` obsługuje submodule zagnieżdżone wewnątrz innych submodule.

---

## 5. Aktualizacja submodule

### Aktualizacja do commitu zapisanego w superprojekcie

```bash
git submodule update --init --recursive
```

Pobiera i przełącza każde submodule na commit wskazany przez superprojekt (zapis w indeksie).

### Pobranie najnowszych zmian z upstream

```bash
git submodule update --remote
```

Aktualizuje submodule do najnowszego commitu ze śledzonej gałęzi (domyślnie `main` lub wartość z `.gitmodules`). Nowy SHA należy zatwierdzić w superprojekcie:

```bash
git add libs/shared-utils
git commit -m "chore: aktualizacja submodule shared-utils do najnowszej wersji"
```

### Aktualizacja wybranego submodule

```bash
git submodule update --remote libs/shared-utils
```

---

## 6. Praca wewnątrz submodule

Submodule po inicjalizacji jest w stanie **detached HEAD** — nie jesteś na żadnej gałęzi. Aby wprowadzać zmiany:

```bash
cd libs/shared-utils
git checkout main          # przejście na gałąź
# ... edycja plików ...
git add .
git commit -m "fix: poprawka w utils"
git push origin main
```

Następnie wróć do superprojektu i zaktualizuj wskaźnik:

```bash
cd ../..
git add libs/shared-utils
git commit -m "chore: aktualizacja shared-utils po poprawce"
```

> **Ważne:** Nie zapomnij wykonać `git push` wewnątrz submodule **przed** commitem w superprojekcie. W przeciwnym razie inni członkowie zespołu nie będą mogli pobrać wskazanego commitu.

---

## 7. Usuwanie submodule

Usunięcie submodule wymaga kilku kroków, ponieważ Git przechowuje jego dane w kilku miejscach.

### Krok 1 — usuń ścieżkę z indeksu

```bash
git submodule deinit -f libs/shared-utils
```

### Krok 2 — usuń katalog z drzewa roboczego

```bash
git rm -f libs/shared-utils
```

### Krok 3 — usuń pozostałości w `.git`

```bash
rm -rf .git/modules/libs/shared-utils
```

### Krok 4 — zatwierdź zmiany

```bash
git commit -m "chore: usunięcie submodule shared-utils"
```

Plik `.gitmodules` zostanie automatycznie zaktualizowany przez `git rm`.

---

## 8. Plik .gitmodules

Plik `.gitmodules` jest tekstowym plikiem konfiguracyjnym w katalogu głównym superprojektu. Przechowuje definicje wszystkich submodule:

```ini
[submodule "libs/shared-utils"]
    path = libs/shared-utils
    url = https://github.com/AI-robot-lab/shared-utils.git
    branch = main

[submodule "data/models"]
    path = data/models
    url = https://github.com/AI-robot-lab/pretrained-models.git
```

**Opis:** Każda sekcja `[submodule "nazwa"]` definiuje jedno submodule. Pole `path` wskazuje katalog w superprojekcie, `url` — zdalne repozytorium, a opcjonalne `branch` — gałąź używaną przy `--remote`.

Plik `.gitmodules` jest częścią repozytorium i należy go commitować oraz udostępniać zespołowi.

---

## 9. Typowe problemy i rozwiązania

### Problem: Pusty katalog submodule po klonowaniu

**Przyczyna:** Klonowanie bez `--recurse-submodules`.

**Rozwiązanie:**

```bash
git submodule update --init --recursive
```

---

### Problem: Submodule wskazuje na nieistniejący commit

**Przyczyna:** Ktoś wypchnął zmiany w submodule, ale nie wypchnął ich na zdalne repozytorium.

**Rozwiązanie:** Wejdź do katalogu submodule i wykonaj `git push`, a następnie zaktualizuj superprojekt.

---

### Problem: Konflikty przy merge dotyczące submodule

**Przyczyna:** Dwie gałęzie superprojektu wskazują na różne commity submodule.

**Rozwiązanie:**

```bash
git checkout --theirs libs/shared-utils   # lub --ours
git submodule update
git add libs/shared-utils
git commit
```

---

### Problem: `detached HEAD` w submodule

**Przyczyna:** Standardowy stan po `git submodule update` — submodule jest przypięty do konkretnego commitu, nie gałęzi.

**Rozwiązanie:** Jeśli chcesz edytować submodule, najpierw przejdź na gałąź:

```bash
cd libs/shared-utils
git checkout main
```

---

### Problem: Zmiana URL submodule

Jeśli repozytorium zewnętrzne zmieniło adres:

```bash
# Edytuj .gitmodules
git submodule sync
git submodule update --init --recursive
```

---

## 10. Dobre praktyki

1. **Zawsze commituj zmianę wskaźnika submodule** po aktualizacji — plik `.gitmodules` i SHA commitu submodule muszą być spójne.

2. **Używaj `--recurse-submodules`** przy klonowaniu i pobieraniu zmian:
   ```bash
   git pull --recurse-submodules
   ```

3. **Komunikuj aktualizacje submodule** w opisie commitu, aby inne osoby wiedziały, że muszą wykonać `git submodule update`.

4. **Nie zmieniaj kodu submodule bezpośrednio w superprojekcie** bez odpowiedniego przygotowania (checkout na gałąź + push).

5. **Rozważ aliasy Git** dla częstych operacji:
   ```bash
   git config --global alias.sup 'submodule update --init --recursive'
   git config --global alias.spull 'pull --recurse-submodules'
   ```

6. **Taguj stabilne wersje submodule** — superprojekt powinien wskazywać na sprawdzone commity, nie na HEAD gałęzi, chyba że świadomie korzystasz z `--remote`.

---

## 11. Ściągawka komend

| Operacja | Komenda |
|---|---|
| Dodanie submodule | `git submodule add <url> <ścieżka>` |
| Klonowanie z submodule | `git clone --recurse-submodules <url>` |
| Inicjalizacja po klonowaniu | `git submodule update --init --recursive` |
| Aktualizacja do wersji z repo | `git submodule update --recursive` |
| Pobranie najnowszej wersji upstream | `git submodule update --remote` |
| Sprawdzenie statusu submodule | `git submodule status` |
| Lista submodule | `git submodule foreach 'echo $name'` |
| Synchronizacja URL | `git submodule sync` |
| Usunięcie submodule | `git submodule deinit -f <ścieżka>` + `git rm -f <ścieżka>` |
| Pull z aktualizacją submodule | `git pull --recurse-submodules` |
