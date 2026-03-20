# Tworzenie nowego repozytorium zadaniowego w organizacji zespołu

Ten artykuł pokazuje nowym członkom zespołu, jak utworzyć nowe repozytorium dla aktualnego zadania wewnątrz wspólnej organizacji GitHub. Przykład opiera się na organizacji `ALF1-RZIT` i zadaniu z bieżącego tygodnia **2026-W12**: „Stworzyć repozytorium dotyczące bieżącej pracy”.

---

## 1. Kiedy tworzyć nowe repozytorium?

Nowe repozytorium warto założyć wtedy, gdy zadanie:

- ma własny kod, dokumentację i historię zmian,
- będzie rozwijane przez więcej niż jedną osobę,
- wymaga osobnego README, issues i pull requestów,
- nie powinno mieszać się z innym aktywnym projektem.

Nie zakładaj nowego repo, jeśli wystarczy:

- nowy branch w istniejącym projekcie,
- nowy katalog w monorepo,
- nowe issue lub milestone.

---

## 2. Przykład aktualnego zadania

Na dzień **2026-03-20** bieżący tydzień roboczy to **2026-W12**, a w planie dla zespołów znajduje się zadanie:

- **„Stworzyć repozytorium dotyczące bieżącej pracy”**.

Dobrym przykładem repozytorium dla takiego zadania może być np. projekt dla zespołu percepcji:

- `percepcja-camera-pipeline-w12`

albo dla zespołu kognicji:

- `kognicja-llm-notes-w12`

Najważniejsze, żeby nazwa była:

- krótka,
- jednoznaczna,
- zgodna z obszarem prac,
- łatwa do wyszukania.

---

## 3. Proponowany schemat nazewnictwa

Polecany format:

```text
<zespol>-<obszar>-<zadanie>-w<tydzien>
```

Przykłady:

- `percepcja-lidar-diagnostics-w12`
- `kognicja-llm-benchmark-w12`
- `interakcja-grasp-demo-w12`

Jeśli repozytorium ma żyć dłużej niż jeden tydzień, można pominąć numer tygodnia i użyć nazwy bardziej produktowej, np.:

- `percepcja-camera-calibration`
- `kognicja-vlm-evaluation`

---

## 4. Tworzenie repozytorium w GitHub Web UI

### 4.1. Przygotowanie

Przed utworzeniem repozytorium ustal:

- właściciela repo (organizacja `ALF1-RZIT`),
- opiekuna technicznego,
- nazwę repozytorium,
- czy repo ma być publiczne czy prywatne,
- początkowy zakres prac.

### 4.2. Kroki

1. Wejdź do organizacji: `https://github.com/ALF1-RZIT`.
2. Kliknij **New** lub **New repository**.
3. Jako właściciela wybierz organizację **ALF1-RZIT**.
4. Wpisz nazwę repozytorium, np. `percepcja-lidar-diagnostics-w12`.
5. Dodaj krótki opis, np. `Eksperymenty i diagnostyka pipeline'u LiDAR dla tygodnia 2026-W12`.
6. Ustaw widoczność:
   - **Private** — jeśli repo zawiera robocze eksperymenty, dane lub konfigurację zespołową,
   - **Public** — jeśli repo ma być otwarte i prezentacyjne.
7. Zaznacz:
   - **Add a README file**,
   - opcjonalnie `.gitignore`,
   - opcjonalnie licencję.
8. Kliknij **Create repository**.

---

## 5. Minimalna zawartość nowego repo

Po utworzeniu repozytorium warto od razu przygotować:

### 5.1. README.md

README powinno odpowiadać na pytania:

- Co to za repo?
- Jaki jest cel zadania?
- Jak uruchomić projekt?
- Kto jest odpowiedzialny?
- Jaki jest aktualny status?

Przykładowy szkielet:

```markdown
# Percepcja LiDAR Diagnostics W12

## Cel
Diagnostyka i testy pipeline'u LiDAR dla robota humanoidalnego.

## Zakres
- logowanie danych,
- analiza opóźnień,
- walidacja timestampów,
- dokumentacja wyników.

## Uruchomienie
Instrukcja pojawi się po przygotowaniu środowiska.

## Zespół
- Imię Nazwisko
- Imię Nazwisko
```

### 5.2. Struktura katalogów

Przykład:

```text
.
├── README.md
├── docs/
├── scripts/
├── data/
├── notebooks/
└── .gitignore
```

### 5.3. Pierwsze issue

Załóż od razu 2–3 issue, np.:

- konfiguracja środowiska,
- pierwszy eksperyment,
- uzupełnienie dokumentacji.

---

## 6. Lokalna inicjalizacja i pierwszy push

Jeśli repo utworzono przez przeglądarkę z README, najprościej je sklonować:

```bash
git clone git@github.com:ALF1-RZIT/percepcja-lidar-diagnostics-w12.git
cd percepcja-lidar-diagnostics-w12
```

Dodaj pierwsze pliki i wykonaj commit:

```bash
mkdir -p docs scripts
printf "# Notatki\n" > docs/notes.md
git add .
git commit -m "Initialize repository structure"
git push origin main
```

Jeśli zaczynasz lokalnie od zera:

```bash
mkdir percepcja-lidar-diagnostics-w12
cd percepcja-lidar-diagnostics-w12
git init
git branch -M main
echo "# Percepcja LiDAR Diagnostics W12" > README.md
git add README.md
git commit -m "Initial commit"
git remote add origin git@github.com:ALF1-RZIT/percepcja-lidar-diagnostics-w12.git
git push -u origin main
```

---

## 7. Nadanie dostępu członkom zespołu

Po utworzeniu repozytorium sprawdź, czy:

- odpowiedni zespół GitHub ma dostęp,
- opiekun techniczny ma uprawnienia administracyjne lub maintain,
- członkowie zespołu mogą tworzyć branche i pull requesty,
- ustawiono podstawowe zasady ochrony brancha `main`, jeśli repo ma większe znaczenie.

---

## 8. Zalecany workflow po utworzeniu repo

1. Utwórz README i podstawową strukturę katalogów.
2. Dodaj pierwsze issues.
3. Ustal konwencję branchy.
4. Otwórz pierwszy PR nawet dla małej zmiany.
5. Dokumentuj wyniki eksperymentów w `docs/`.
6. Po zakończeniu tygodnia zrób krótkie podsumowanie statusu.

---

## 9. Najczęstsze błędy

- nazwa repo jest zbyt ogólna, np. `projekt1`,
- brak README i celu projektu,
- wrzucanie danych tymczasowych i dużych artefaktów bez `.gitignore`,
- praca bez issues i bez właściciela zadania,
- commitowanie bezpośrednio do `main`,
- mieszanie kilku niezależnych zadań w jednym repo.

---

## 10. Checklista dla nowej osoby

- [ ] Mam dostęp do organizacji `ALF1-RZIT`.
- [ ] Uzgodniłem nazwę i cel repozytorium.
- [ ] Dodałem README.
- [ ] Dodałem `.gitignore`.
- [ ] Wykonałem pierwszy commit.
- [ ] Sprawdziłem uprawnienia zespołu.
- [ ] Założyłem pierwsze issue.

---

## Powiązane artykuły

- [Praca w zespole inżynierskim](#wiki-praca-w-zespole)
- [Git i GitHub](#wiki-git-github)
- [GitHub Releases i Packages](#wiki-github_releases_packages)

## Zasoby

- GitHub organization: https://github.com/ALF1-RZIT
- GitHub Docs: creating repositories in organizations
- Wewnętrzny plan zadań zespołowych

---
*Ostatnia aktualizacja: 2026-03-20*
*Autor: Codex / Zespół Laboratorium*
