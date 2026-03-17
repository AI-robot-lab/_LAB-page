# Git i GitHub — kompleksowy poradnik

Niniejszy przewodnik omawia kontrolę wersji od podstaw: czym jest Git, jak
działa, jakie są kluczowe komendy, jak wygląda praca grupowa na GitHubie oraz
jakich zasad i dobrych praktyk należy przestrzegać na co dzień.

---

## Spis treści

1. [Idea kontroli wersji](#1-idea-kontroli-wersji)
2. [Jak działa Git](#2-jak-działa-git)
   1. [Trzy obszary robocze](#21-trzy-obszary-robocze)
   2. [Obiektowy model danych](#22-obiektowy-model-danych)
3. [Instalacja i konfiguracja](#3-instalacja-i-konfiguracja)
4. [Podstawowe komendy](#4-podstawowe-komendy)
   1. [Inicjalizacja i klonowanie](#41-inicjalizacja-i-klonowanie)
   2. [Rejestrowanie zmian](#42-rejestrowanie-zmian)
   3. [Historia i inspekcja](#43-historia-i-inspekcja)
   4. [Gałęzie](#44-gałęzie)
   5. [Scalanie i rebase](#45-scalanie-i-rebase)
   6. [Praca ze zdalnym repozytorium](#46-praca-ze-zdalnym-repozytorium)
   7. [Cofanie zmian](#47-cofanie-zmian)
5. [Praktyczne przykłady](#5-praktyczne-przykłady)
   1. [Nowy projekt od zera](#51-nowy-projekt-od-zera)
   2. [Nowa funkcja w osobnej gałęzi](#52-nowa-funkcja-w-osobnej-gałęzi)
   3. [Hotfix na produkcji](#53-hotfix-na-produkcji)
   4. [Interaktywny rebase — porządkowanie historii](#54-interaktywny-rebase--porządkowanie-historii)
   5. [Stash — odkładanie zmian na później](#55-stash--odkładanie-zmian-na-później)
   6. [Cherry-pick — przenoszenie wybranych commitów](#56-cherry-pick--przenoszenie-wybranych-commitów)
6. [Praca grupowa na GitHubie](#6-praca-grupowa-na-githubie)
   1. [Model Fork & Pull Request](#61-model-fork--pull-request)
   2. [Model Feature Branch (Shared Repository)](#62-model-feature-branch-shared-repository)
   3. [Tworzenie Pull Requesta krok po kroku](#63-tworzenie-pull-requesta-krok-po-kroku)
   4. [Code review — zasady i etykieta](#64-code-review--zasady-i-etykieta)
   5. [Rozwiązywanie konfliktów scalania](#65-rozwiązywanie-konfliktów-scalania)
7. [Strategie rozgałęzień](#7-strategie-rozgałęzień)
   1. [Git Flow](#71-git-flow)
   2. [GitHub Flow](#72-github-flow)
   3. [Trunk-Based Development](#73-trunk-based-development)
8. [Uniwersalne zasady i dobre praktyki](#8-uniwersalne-zasady-i-dobre-praktyki)
9. [Praktyczne porady na co dzień](#9-praktyczne-porady-na-co-dzień)
10. [Ściągawka komend](#10-ściągawka-komend)

---

## 1. Idea kontroli wersji

**Kontrola wersji** (ang. *version control* lub *source control*) to system
rejestrujący zmiany plików w czasie, umożliwiający przywrócenie dowolnego
wcześniejszego stanu projektu.

### Dlaczego kontrola wersji jest niezbędna?

| Problem bez kontroli wersji | Rozwiązanie z kontrolą wersji |
|---|---|
| Przypadkowe usunięcie lub nadpisanie pliku | Pełna historia zmian; powrót do każdej wersji |
| Brak wiedzy, kto i kiedy wprowadził zmianę | Każdy commit ma autora, datę i opis |
| Równoczesna praca kilku osób nad tym samym plikiem | Mechanizmy scalania (merge) i rozwiązywania konfliktów |
| „Backup" przez kopiowanie katalogów (`projekt_v2_final_FINAL`) | Jedno repozytorium z pełną historią |
| Niemożność eksperymentowania bez ryzyka | Gałęzie (branches) izolują eksperymenty |

### Git vs. centralne systemy kontroli wersji

Git jest **rozproszonym** systemem kontroli wersji (DVCS). Każdy programista
posiada lokalne, pełne repozytorium z całą historią projektu. Pozwala to
pracować offline i znacznie przyspiesza większość operacji.

```
Centralny SVN/CVS            Rozproszony Git
─────────────────            ───────────────────────────
      Serwer                        GitHub/GitLab
      │                      ┌──────────┴──────────┐
  ┌───┴───┐               Klon A              Klon B
  │       │             (pełna historia)   (pełna historia)
Klient A Klient B
```

---

## 2. Jak działa Git

### 2.1. Trzy obszary robocze

Git zarządza plikami w trzech przestrzeniach:

```
Working Directory    Staging Area (Index)    Repository (.git)
─────────────────    ────────────────────    ────────────────
  Pliki na dysku   ──git add──▶  Snapshot  ──git commit──▶  Historia
                   ◀─git restore─          ◀─git checkout─
```

| Obszar | Opis |
|---|---|
| **Working Directory** | Zwykłe pliki w folderze projektu, które edytujesz |
| **Staging Area** | Przejściowy obszar; wybierasz tu, co trafi do następnego commita |
| **Repository** | Baza danych Gita (folder `.git`); niezmienne snapshoty (commity) |

### 2.2. Obiektowy model danych

Git przechowuje dane jako cztery typy obiektów identyfikowanych przez SHA-1/SHA-256:

| Typ | Opis |
|---|---|
| **blob** | Zawartość pojedynczego pliku |
| **tree** | Katalog — lista blob-ów i innych tree-ów z nazwami |
| **commit** | Snapshot projektu; wskazuje na tree, autora, datę i rodzica |
| **tag** | Oznaczona etykieta wskazująca konkretny commit |

```
commit 3f9a1b2
├── author: Anna Kowalski
├── date: 2025-01-15
├── message: "Add robot arm controller"
├── parent: a1b2c3d
└── tree: 7e8f9a0
    ├── src/ (tree)
    │   ├── controller.py (blob)
    │   └── utils.py (blob)
    └── README.md (blob)
```

---

## 3. Instalacja i konfiguracja

### Instalacja

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install git

# macOS (Homebrew)
brew install git

# Windows
# Pobierz instalator z https://git-scm.com/download/win
```

### Konfiguracja globalna (jednorazowa)

```bash
# Tożsamość — pojawi się w każdym commicie
git config --global user.name "Anna Kowalski"
git config --global user.email "anna@example.com"

# Domyślny edytor tekstu
git config --global core.editor "code --wait"   # VS Code
git config --global core.editor "nano"          # Nano

# Domyślna nazwa gałęzi głównej
git config --global init.defaultBranch main

# Wyświetlanie konfiguracji
git config --global --list
```

### Klucze SSH do GitHuba (zalecane)

```bash
# Wygeneruj parę kluczy
ssh-keygen -t ed25519 -C "anna@example.com"

# Wyświetl klucz publiczny i wklej go na GitHub
# (Settings → SSH and GPG keys → New SSH key)
cat ~/.ssh/id_ed25519.pub
```

---

## 4. Podstawowe komendy

### 4.1. Inicjalizacja i klonowanie

```bash
# Nowe repozytorium w bieżącym folderze
git init

# Klonowanie istniejącego repozytorium
git clone https://github.com/user/repo.git

# Klonowanie przez SSH (wymaga klucza SSH)
git clone git@github.com:user/repo.git

# Klonowanie do konkretnego folderu
git clone https://github.com/user/repo.git moj-folder
```

### 4.2. Rejestrowanie zmian

```bash
# Sprawdź status — co jest zmienione, co w staging area
git status

# Dodaj plik do staging area
git add plik.py

# Dodaj wszystkie zmienione pliki
git add .

# Dodaj fragmenty pliku interaktywnie
git add -p plik.py

# Utwórz commit z krótką wiadomością
git commit -m "Add ROS2 node for arm controller"

# Dodaj do staging area i commituj w jednym kroku (tylko śledzone pliki)
git commit -am "Fix typo in README"

# Popraw ostatni commit (zmień wiadomość lub dodaj zapomniane pliki)
git add zapomniane.py
git commit --amend --no-edit
```

### 4.3. Historia i inspekcja

```bash
# Pełna historia commitów
git log

# Skrócony widok (jedna linia na commit)
git log --oneline

# Graf gałęzi w terminalu
git log --oneline --graph --all

# Pokaż zmiany w ostatnim commicie
git show

# Pokaż zmiany w konkretnym commicie
git show abc1234

# Porównaj working directory ze staging area
git diff

# Porównaj staging area z ostatnim commitem
git diff --staged

# Porównaj dwie gałęzie
git diff main..feature/arm-controller

# Kto zmienił każdą linię pliku
git blame plik.py

# Znajdź commit, który wprowadził błąd (binarne przeszukiwanie)
git bisect start
git bisect bad                  # obecny commit jest zepsuty
git bisect good v1.0            # ta wersja była dobra
# Git wychodzi do środkowego commitu — testuj i oznaczaj:
git bisect good / git bisect bad
git bisect reset                # koniec
```

### 4.4. Gałęzie

```bash
# Lista gałęzi lokalnych
git branch

# Lista gałęzi zdalnych
git branch -r

# Lista wszystkich gałęzi
git branch -a

# Utwórz nową gałąź
git branch feature/lidar-mapping

# Przełącz na gałąź
git checkout feature/lidar-mapping

# Utwórz i przełącz w jednym kroku (nowszy styl)
git switch -c feature/lidar-mapping

# Usuń gałąź (lokalnie)
git branch -d feature/lidar-mapping   # bezpieczne — tylko scalona
git branch -D feature/lidar-mapping   # wymuś usunięcie

# Usuń gałąź zdalną
git push origin --delete feature/lidar-mapping

# Zmień nazwę bieżącej gałęzi
git branch -m nowa-nazwa
```

### 4.5. Scalanie i rebase

```bash
# Scal gałąź feature do bieżącej (np. main)
git checkout main
git merge feature/lidar-mapping

# Scal z zachowaniem merge commita (--no-ff)
git merge --no-ff feature/lidar-mapping

# Rebase — przenieś commity feature na czubek main
git checkout feature/lidar-mapping
git rebase main

# Interaktywny rebase — edytuj ostatnie N commitów
git rebase -i HEAD~3

# Przerwij rebase w trakcie konfliktu
git rebase --abort

# Kontynuuj rebase po rozwiązaniu konfliktu
git rebase --continue
```

### 4.6. Praca ze zdalnym repozytorium

```bash
# Lista zdalnych repozytoriów
git remote -v

# Dodaj zdalne repozytorium
git remote add origin git@github.com:user/repo.git

# Pobierz zmiany bez scalania
git fetch origin

# Pobierz i scal (fetch + merge)
git pull origin main

# Wyślij zmiany na serwer
git push origin main

# Wyślij nową gałąź
git push -u origin feature/lidar-mapping

# Po ustawieniu upstream: skrót
git push

# Wymuś push (ostrożnie! nadpisuje historię zdalną)
git push --force-with-lease   # bezpieczniejszy wariant
```

### 4.7. Cofanie zmian

```bash
# Cofnij niestage'owane zmiany w pliku
git restore plik.py

# Cofnij stage'owanie pliku (wróć do working directory)
git restore --staged plik.py

# Cofnij zmianę przez nowy commit (bezpieczne — zachowuje historię)
git revert abc1234

# Przenieś HEAD i branch o N commitów wstecz
git reset --soft  HEAD~1    # zachowaj zmiany w staging area
git reset --mixed HEAD~1    # zachowaj zmiany w working dir (domyślne)
git reset --hard  HEAD~1    # ZNISZCZ zmiany (nieodwracalne!)

# Odtwórz usuniętą gałąź lub commit z reflog
git reflog
git checkout -b odzyskana-galaz abc1234
```

---

## 5. Praktyczne przykłady

### 5.1. Nowy projekt od zera

```bash
mkdir robot-arm-controller && cd robot-arm-controller
git init

# Utwórz plik .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.env
*.log
EOF

# Pierwszy commit
echo "# Robot Arm Controller" > README.md
git add .
git commit -m "Initial commit: project structure"

# Połącz z GitHubem
git remote add origin git@github.com:team/robot-arm-controller.git
git push -u origin main
```

### 5.2. Nowa funkcja w osobnej gałęzi

```bash
# Zaktualizuj main przed odgałęzieniem
git switch main
git pull

# Stwórz gałąź dla nowej funkcji
git switch -c feature/gripper-control

# Pracuj, commituj małymi krokami
echo "class Gripper: ..." > gripper.py
git add gripper.py
git commit -m "Add Gripper class skeleton"

# ... więcej pracy ...
git commit -m "Implement open/close methods"
git commit -m "Add force feedback support"

# Wypchnij gałąź na GitHub
git push -u origin feature/gripper-control

# Otwórz Pull Request na GitHubie, poczekaj na review
# Po zatwierdzeniu — scal do main przez interfejs GitHub lub:
git switch main
git merge --no-ff feature/gripper-control
git push

# Posprzątaj
git branch -d feature/gripper-control
git push origin --delete feature/gripper-control
```

### 5.3. Hotfix na produkcji

```bash
# Odgałęź od tagu produkcyjnego
git switch -c hotfix/fix-gripper-crash v2.1.0

# Napraw błąd
vim gripper.py
git commit -am "Fix: prevent gripper crash on timeout"

# Scal do main i oznacz nową wersją
git switch main
git merge --no-ff hotfix/fix-gripper-crash
git tag -a v2.1.1 -m "Hotfix: gripper crash fix"
git push && git push --tags

# Scal też do gałęzi develop (jeśli używasz Git Flow)
git switch develop
git merge --no-ff hotfix/fix-gripper-crash
git push

# Usuń gałąź hotfix
git branch -d hotfix/fix-gripper-crash
```

### 5.4. Interaktywny rebase — porządkowanie historii

Przed wysłaniem Pull Requesta warto oczyścić historię commitów:

```bash
git rebase -i HEAD~4
```

Otworzy się edytor z listą ostatnich 4 commitów:

```
pick abc1234 Add Gripper class skeleton
pick def5678 WIP: half-done open method
pick ghi9012 Fix typo
pick jkl3456 Implement open/close methods

# Komendy:
# p, pick   — zachowaj commit
# r, reword — zachowaj, ale zmień wiadomość
# s, squash — połącz z poprzednim commitem
# f, fixup  — połącz z poprzednim (porzuć wiadomość)
# d, drop   — usuń commit
```

Zmień na:

```
pick abc1234 Add Gripper class skeleton
fixup def5678 WIP: half-done open method
fixup ghi9012 Fix typo
reword jkl3456 Implement open/close methods with force feedback
```

Wynik: dwa czyste commity zamiast czterech roboczych.

### 5.5. Stash — odkładanie zmian na później

```bash
# Odłóż bieżące zmiany (łącznie z nienazwanymi)
git stash push -m "WIP: gripper calibration"

# Przełącz na inną gałąź, napraw pilny błąd, wróć
git switch main
# ... naprawki ...
git switch feature/gripper-control

# Przywróć odłożone zmiany
git stash pop

# Lista schowka
git stash list

# Zastosuj konkretny wpis ze schowka
git stash apply stash@{2}
```

### 5.6. Cherry-pick — przenoszenie wybranych commitów

```bash
# Przenieś konkretny commit z innej gałęzi
git switch main
git cherry-pick abc1234

# Przenieś zakres commitów
git cherry-pick abc1234^..def5678
```

---

## 6. Praca grupowa na GitHubie

### 6.1. Model Fork & Pull Request

Stosowany w projektach open-source, gdy zewnętrzni współtwórcy nie mają
praw zapisu do głównego repozytorium.

```
1. Fork         GitHub: utwórz fork (kopia pod własnym kontem)
2. Clone        git clone git@github.com:TY/repo.git
3. Upstream     git remote add upstream git@github.com:ORG/repo.git
4. Branch       git switch -c fix/docs-typo
5. Commit       git commit -m "Fix typo in README"
6. Push         git push origin fix/docs-typo
7. Pull Request Utwórz PR na GitHubie: TY:fix/docs-typo → ORG:main
8. Review       Otrzymaj komentarze, wprowadź poprawki, push
9. Merge        Opiekun projektu scala PR
```

Synchronizacja forka z oryginałem:

```bash
git fetch upstream
git switch main
git merge upstream/main
git push origin main
```

### 6.2. Model Feature Branch (Shared Repository)

Stosowany w zamkniętych zespołach, gdzie wszyscy mają dostęp do jednego
repozytorium.

```
main ────●─────────────────────●─── (tylko przez PRy)
          \                   /
feature    ●─●─●─●──●────────●
```

Zasady:
- **Nigdy** nie commituj bezpośrednio do `main`/`master`.
- Każda zmiana (funkcja, bugfix, refaktor) = osobna gałąź.
- Gałąź żyje krótko — maksymalnie kilka dni.
- PR wymaga co najmniej jednej recenzji przed scaleniem.

### 6.3. Tworzenie Pull Requesta krok po kroku

```bash
# 1. Zaktualizuj gałąź bazową
git switch main && git pull

# 2. Utwórz gałąź
git switch -c feature/slam-improvements

# 3. Praca i commity
git commit -m "Improve SLAM loop closure detection"

# 4. Zaktualizuj o ewentualne zmiany z main (rebase lub merge)
git fetch origin
git rebase origin/main

# 5. Wypchnij
git push -u origin feature/slam-improvements
```

Na GitHubie:
1. Otwórz repozytorium → kliknij **Compare & pull request**.
2. Wypełnij tytuł i opis (co, dlaczego, jak przetestować).
3. Przypisz recenzentów (Reviewers) i etykiety (Labels).
4. Poczekaj na CI/CD (automatyczne testy).
5. Odpowiedz na komentarze recenzentów — pushuj poprawki.
6. Po zatwierdzeniu — kliknij **Merge pull request**.

### 6.4. Code review — zasady i etykieta

**Dla recenzenta:**
- Komentuj konkretnie i konstruktywnie — wskazuj, co i dlaczego.
- Odróżniaj blokery (`Blokuje merge: ...`) od sugestii (`Nit: ...`).
- Przeglądaj zmianę jako całość, nie tylko linijkę po linijce.
- Zatwierdzaj (Approve) tylko kod, który rozumiesz i który działa.

**Dla autora:**
- Odpowiadaj na każdy komentarz — nawet jeśli nie zgadzasz się, wyjaśnij dlaczego.
- Nie commituj dodatkowych zmian w trakcie aktywnego review.
- Resolve komentarze dopiero po ich rozwiązaniu — nie zamykaj dyskusji bez odpowiedzi.

### 6.5. Rozwiązywanie konfliktów scalania

Konflikt pojawia się, gdy dwie osoby zmodyfikowały te same linie pliku.

```bash
# Podczas merge/rebase Git zatrzymuje się przy konflikcie
git merge feature/slam-improvements

# Status pokazuje pliki z konfliktami
git status
# both modified: slam/mapper.cpp

# Otwórz plik — znajdziesz markery konfliktu
```

```cpp
<<<<<<< HEAD (twoja wersja — main)
float loop_threshold = 0.85f;
=======
float loop_threshold = 0.90f;
>>>>>>> feature/slam-improvements (przychodzące zmiany)
```

```bash
# Edytuj plik ręcznie lub użyj narzędzia
git mergetool        # np. meld, vimdiff, VS Code

# Po rozwiązaniu: dodaj plik i kontynuuj
git add slam/mapper.cpp
git merge --continue   # lub: git commit
```

---

## 7. Strategie rozgałęzień

### 7.1. Git Flow

Klasyczny model dla projektów z regularnymi wydaniami.

```
main      ●────────────────────────────●───── (produkcja)
           \                          / \
hotfix      ●──●                     /   ●──●
                \                   /
release          ●──●──────────────●
                /
develop   ●────●──────────────────────────────
          \       \           /
feature    ●──●    ●──●──●──●
```

| Gałąź | Opis |
|---|---|
| `main` | Kod na produkcji; tylko przez release lub hotfix |
| `develop` | Gałąź integracyjna; tutaj zbiegają się feature'y |
| `feature/*` | Nowe funkcje; odgałęzia się od `develop` |
| `release/*` | Przygotowanie wydania; bugfixy, bumping wersji |
| `hotfix/*` | Pilne poprawki bezpośrednio z `main` |

### 7.2. GitHub Flow

Prostszy model dla ciągłego wdrażania (CI/CD).

```
main ────●──────●──────●──────●──────
          \    / \    / \    /
branch     ●──●   ●──●   ●──●
```

Zasada: `main` jest zawsze stabilny i wdrażalny. Każda zmiana trafia przez PR.

### 7.3. Trunk-Based Development

Wszyscy pracują na jednej gałęzi (`main`/`trunk`). Stosowane przy bardzo
dojrzałym CI/CD i małych, częstych commitach. Funkcje ukrywane przez
*feature flags*.

---

## 8. Uniwersalne zasady i dobre praktyki

### Commity

1. **Jeden commit = jedna logiczna zmiana.** Nie miksuj refactoringu
   z bugfixem w jednym commicie.
2. **Konwencja wiadomości (Conventional Commits):**

   ```
   <typ>(<zakres>): <krótki opis>

   [opcjonalny rozwinięty opis]

   [opcjonalne: BREAKING CHANGE lub referencje do Issues]
   ```

   Typy: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `ci`.

   Przykłady:
   ```
   feat(slam): add loop closure detection
   fix(gripper): prevent crash on timeout — fixes #42
   docs(readme): update installation instructions
   refactor(kinematics): extract FK solver to separate module
   ```

3. **Pisz wiadomości w trybie rozkazującym:** „Add feature" zamiast
   „Added feature" lub „Adding feature".
4. **Nie commituj kodu, który nie kompiluje / psuje testy.**
5. **Nie commituj sekretów** (API keys, hasła, tokeny). Używaj `.gitignore`
   i zmiennych środowiskowych.

### Gałęzie

6. **Nazwij gałąź opisowo:** `feature/lidar-mapping`, `fix/imu-drift`,
   `docs/update-api-reference`.
7. **Gałęzie powinny być krótkotrwałe** — merguj jak najszybciej, by
   unikać długich conflictów.
8. **Gałąź główna jest zawsze stabilna** — nikt nie pushuje bezpośrednio.

### Zespołowa praca

9.  **Pull przed push** — zawsze aktualizuj lokalną kopię przed wysłaniem zmian.
10. **Komunikuj się przez PR** — opis, komentarze i recenzje są dokumentacją
    decyzji projektowych.
11. **Używaj Issues i Milestones** do śledzenia pracy; linkuj PR do Issue
    (`fixes #42`).
12. **CI/CD jest obowiązkowe** — żaden PR nie może być scalony bez zielonych
    testów automatycznych.

### Bezpieczeństwo

13. **Nigdy nie używaj `git push --force` na gałęziach współdzielonych.**
    Użyj `--force-with-lease` jeśli musisz.
14. **Nie przepisuj publicznej historii** (rebase/amend) po tym, jak ktoś
    inny oparł na niej swoje commity.
15. **.gitignore** — wersjonuj go od pierwszego commita. Szablony:
    [gitignore.io](https://gitignore.io).

---

## 9. Praktyczne porady na co dzień

### Aliasy Git — przyspiesz pracę

```bash
git config --global alias.st "status"
git config --global alias.co "checkout"
git config --global alias.br "branch"
git config --global alias.lg "log --oneline --graph --all --decorate"
git config --global alias.last "log -1 HEAD"
git config --global alias.unstage "restore --staged"
```

Teraz `git lg` wyświetla czytelne drzewo historii, a `git st` zastępuje
`git status`.

### Automatyczne poprawne zakończenia linii

```bash
# macOS / Linux
git config --global core.autocrlf input

# Windows
git config --global core.autocrlf true
```

### .gitignore — globalne wzorce

```bash
# Utwórz globalny .gitignore (IDE, systemy operacyjne)
cat > ~/.gitignore_global << 'EOF'
.DS_Store
Thumbs.db
.idea/
.vscode/
*.swp
*~
EOF
git config --global core.excludesFile ~/.gitignore_global
```

### Sprawdzanie co się zmieniło przed commitem

```bash
# Podsumowanie zmian
git diff --stat

# Wizualne diff w VS Code
git difftool --tool=vscode
```

### Szybkie cofanie błędnych operacji

```bash
# Właśnie zrobiłem błędny commit — cofnij, zachowaj zmiany
git reset --soft HEAD~1

# Zepsuty merge — wróć do stanu sprzed
git merge --abort

# Zgubiłem commit po reset --hard — sprawdź reflog
git reflog
git checkout -b ratunkowa-galaz abc1234
```

### Dobre nawyki

- Commituj **często i małymi krokami** — łatwiej review, łatwiej cofnąć.
- Przed każdym `push` zrób `git log --oneline` i upewnij się, że historia
  wygląda sensownie.
- Regularnie synchronizuj gałęź feature z `main`/`develop` (`rebase` lub
  `merge`), żeby unikać długich, bolesnych conflictów.
- Używaj **branchy dla każdej zmiany** — nawet jednolinijkowego bugfixa.
- Opisuj PR tak, jakbyś tłumaczył zmianę osobie, która nie zna kontekstu —
  bo za pół roku to będziesz Ty.

### Przydatne narzędzia graficzne

| Narzędzie | Platforma | Opis |
|---|---|---|
| **GitHub Desktop** | Win/Mac | Prosty klient dla początkujących |
| **GitKraken** | Win/Mac/Lin | Wizualne drzewo gałęzi, wbudowany review |
| **Sourcetree** | Win/Mac | Darmowy, rozbudowany |
| **VS Code** (wbudowany) | Wszystkie | Source Control + GitLens rozszerzenie |
| **lazygit** | TUI | Szybki terminal UI, polecany dla zaawansowanych |

---

## 10. Ściągawka komend

| Zadanie | Komenda |
|---|---|
| Inicjalizacja repozytorium | `git init` |
| Klonowanie | `git clone <url>` |
| Status plików | `git status` |
| Dodaj do staging | `git add <plik>` / `git add .` |
| Commit | `git commit -m "opis"` |
| Historia | `git log --oneline --graph --all` |
| Nowa gałąź | `git switch -c <nazwa>` |
| Zmień gałąź | `git switch <nazwa>` |
| Scal gałąź | `git merge <nazwa>` |
| Rebase | `git rebase <baza>` |
| Pobierz zmiany | `git pull` |
| Wyślij zmiany | `git push` |
| Odłóż zmiany | `git stash push -m "opis"` |
| Przywróć stash | `git stash pop` |
| Cofnij zmianę w pliku | `git restore <plik>` |
| Cofnij ostatni commit | `git reset --soft HEAD~1` |
| Usuń zmiany (!)| `git reset --hard HEAD` |
| Revert commitu | `git revert <sha>` |
| Tagi | `git tag -a v1.0 -m "opis"` |
| Wyślij tagi | `git push --tags` |
| Szukaj błędu | `git bisect start/good/bad` |
| Historia zmian linii | `git blame <plik>` |

---

> **Wskazówka końcowa:** Git to narzędzie, które nagradza regularną,
> zdyscyplinowaną pracę. Małe, dobrze opisane commity, krótkotrwałe gałęzie
> i rzetelne code review sprawiają, że historia projektu staje się
> bezcenną dokumentacją — a nie chaosem, przez który nikt nie chce przebijać.
