# GitHub Releases — czym są i jak je tworzyć

GitHub Releases to mechanizm publikowania uporządkowanych wersji projektu na bazie tagów Git. Dla zespołu inżynierskiego release jest punktem kontrolnym: pozwala wskazać konkretną wersję kodu, opisać zmiany i dołączyć gotowe artefakty, np. archiwa, pliki binarne lub dokumentację.

---

## 1. Czym jest GitHub Release?

Release na GitHubie składa się zwykle z kilku elementów:

- **tagu Git** wskazującego konkretny commit,
- **nazwy wydania**, np. `v1.2.0`,
- **opisu zmian** (*release notes*),
- **załączników** (*assets*), np. plików `.zip`, `.tar.gz`, firmware'u lub PDF,
- opcjonalnego oznaczenia jako **pre-release** albo **latest**.

Najprościej mówiąc:

- **tag** odpowiada na pytanie: *który commit oznacza tę wersję?*
- **release** odpowiada na pytanie: *jak przedstawić tę wersję ludziom i co mają z niej pobrać?*

---

## 2. Po co zespołowi GitHub Releases?

GitHub Releases są przydatne, gdy chcesz:

- publikować stabilne wersje projektu,
- przekazywać wyniki pracy prowadzącemu, opiekunowi lub innemu zespołowi,
- archiwizować wersje demonstracyjne przed pokazem,
- udostępniać gotowe paczki bez konieczności klonowania repo,
- prowadzić czytelną historię rozwoju projektu.

W praktyce release dobrze sprawdza się np. dla:

- wersji demonstracyjnej modułu percepcji,
- paczki konfiguracyjnej dla Jetsona,
- archiwum wyników eksperymentu,
- stabilnej wersji kodu przed integracją na robocie.

---

## 3. Jak działa release w relacji do Git?

Najpierw w repozytorium powstaje commit, później tag, a dopiero potem release.

```text
commit -> tag -> GitHub Release -> assets + opis zmian
```

**Opis kodu:** Ten prosty schemat pokazuje kolejność działań. Najpierw zatwierdzasz kod w repozytorium, następnie oznaczasz konkretny commit tagiem, a dopiero na końcu tworzysz release na GitHubie, który wykorzystuje ten tag jako podstawę wydania.

---

## 4. Kiedy tworzyć release?

Release warto utworzyć, gdy:

- zakończyłeś ważny etap prac,
- masz wersję gotową do demonstracji,
- chcesz przekazać działającą wersję innemu zespołowi,
- wypuszczasz wersję testową dla użytkowników,
- zamykasz sprint, milestone albo tydzień projektowy.

Nie twórz release po każdej drobnej zmianie dokumentacji. Release powinien oznaczać sensowny, rozpoznawalny stan projektu.

---

## 5. Nazewnictwo wersji

Najczęściej stosuje się **SemVer** (*Semantic Versioning*):

- `v1.0.0` — pierwsza stabilna wersja,
- `v1.1.0` — nowa funkcja bez łamania kompatybilności,
- `v1.1.1` — poprawka błędu,
- `v2.0.0` — duża zmiana lub breaking change.

Dla wersji testowych można używać np.:

- `v1.2.0-beta.1`,
- `v1.2.0-rc.1`.

---

## 6. Tworzenie release w interfejsie GitHub

To najprostsza metoda dla nowych członków zespołu.

### Kroki

1. Wejdź do repozytorium.
2. Otwórz zakładkę **Releases**.
3. Kliknij **Draft a new release**.
4. Wybierz istniejący tag albo utwórz nowy.
5. Podaj tytuł wydania.
6. Uzupełnij opis zmian.
7. Dodaj pliki do pobrania, jeśli są potrzebne.
8. Wybierz, czy to ma być `pre-release`.
9. Kliknij **Publish release**.

### Co wpisać w opisie release?

Dobry opis release zawiera:

- cel wydania,
- najważniejsze nowe funkcje,
- poprawki błędów,
- znane ograniczenia,
- instrukcję użycia lub migracji.

Przykład struktury notatek:

```markdown
## Co nowego
- dodano obsługę synchronizacji danych LiDAR + kamera
- poprawiono logowanie diagnostyczne na Jetsonie

## Poprawki
- naprawiono błąd parsowania timestampów

## Znane ograniczenia
- testowano tylko na konfiguracji laboratoryjnej
```

**Opis kodu:** Ten blok pokazuje przykładową strukturę `release notes` w Markdown. Sekcja „Co nowego” służy do opisania funkcji, „Poprawki” do zgłoszonych błędów, a „Znane ograniczenia” do uczciwego wskazania, czego jeszcze nie zweryfikowano.

---

## 7. Tworzenie release z terminala

### 7.1. Utworzenie tagu Git

Najpierw oznacz commit tagiem z adnotacją.

```bash
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

**Opis kodu:** Pierwsza komenda tworzy tag `v1.0.0` i zapisuje do niego komentarz opisujący wydanie. Druga komenda wysyła ten tag do zdalnego repozytorium, aby GitHub mógł go wykorzystać przy tworzeniu release.

### 7.2. Utworzenie release przez GitHub CLI

Jeśli w zespole używacie `gh`, release można utworzyć bezpośrednio z terminala.

```bash
gh release create v1.0.0 \
  --title "Wersja v1.0.0" \
  --notes "Stabilne wydanie modułu percepcji" \
  ./dist/demo.zip ./dist/docs.pdf
```

**Opis kodu:** Komenda `gh release create` tworzy release dla istniejącego tagu `v1.0.0`. Parametr `--title` ustawia nazwę widoczną na GitHubie, `--notes` dodaje opis zmian, a na końcu podajesz pliki, które mają zostać dołączone do wydania jako assets.

### 7.3. Automatyczne wygenerowanie notatek

GitHub może sam przygotować wstępny opis na podstawie pull requestów i commitów.

```bash
gh release create v1.0.1 --generate-notes
```

**Opis kodu:** Ta komenda tworzy release dla tagu `v1.0.1` i każe GitHubowi automatycznie wygenerować notatki wydania. To dobre rozwiązanie, gdy chcesz szybko opublikować wersję, ale nadal warto później sprawdzić, czy wygenerowany opis jest czytelny.

### 7.4. Wersja testowa (pre-release)

Jeśli wydanie nie jest jeszcze produkcyjne, oznacz je jako testowe.

```bash
gh release create v1.1.0-beta.1 \
  --prerelease \
  --generate-notes
```

**Opis kodu:** Tutaj tworzony jest release oznaczony jako `pre-release`, czyli wersja testowa. Dzięki temu użytkownicy widzą, że to nie jest jeszcze finalne wydanie stabilne.

---

## 8. Automatyzacja release przez GitHub Actions

W bardziej dojrzałych projektach release może tworzyć się automatycznie po wypchnięciu tagu.

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: write

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Publish release
        uses: softprops/action-gh-release@v2
        with:
          generate_release_notes: true
          files: |
            dist/*.zip
            dist/*.pdf
```

**Opis kodu:** Ten workflow uruchamia się tylko wtedy, gdy do repozytorium zostanie wypchnięty tag zaczynający się od `v`, np. `v1.0.0`. Najpierw pobiera pełną historię repo, a potem publikuje release przy użyciu gotowej akcji `softprops/action-gh-release`, dołączając wskazane pliki z katalogu `dist/`.

Jeśli chcesz uruchomić taki workflow, typowy przebieg wygląda tak:

```bash
git tag -a v1.2.0 -m "Release v1.2.0"
git push origin v1.2.0
```

**Opis kodu:** Ten zestaw komend nie tworzy release bezpośrednio sam z siebie — jego zadaniem jest uruchomienie workflow GitHub Actions przez wypchnięcie tagu do zdalnego repozytorium. To dobry model pracy, gdy chcesz, aby publikacja była zautomatyzowana i powtarzalna.

---

## 9. Co dołączać do release?

Do release można dodać:

- skompilowane paczki `.zip` lub `.tar.gz`,
- pliki firmware,
- PDF z instrukcją uruchomienia,
- konfiguracje testowe,
- raport z eksperymentu.

Nie warto dodawać:

- sekretów,
- dużych danych surowych bez uzasadnienia,
- plików tymczasowych,
- artefaktów, których nikt nie potrafi zidentyfikować.

Dobrą praktyką jest nazwanie plików jasno, np.:

```text
robot-perception-v1.0.0.zip
robot-perception-v1.0.0-manual.pdf
robot-perception-v1.0.0-checksum.txt
```

**Opis kodu:** Ten przykład pokazuje czytelne nazwy plików dołączanych do release. W nazwie warto zawrzeć nazwę projektu, numer wersji i typ pliku, aby odbiorca od razu wiedział, co pobiera.

---

## 10. Dobre praktyki

1. Twórz release tylko dla sensownych punktów kontrolnych projektu.
2. Używaj tagów z adnotacją, a nie przypadkowych lekkich tagów.
3. Stosuj spójne nazewnictwo wersji.
4. Zawsze opisuj zmiany z perspektywy użytkownika zespołu.
5. Oznaczaj wersje testowe jako `pre-release`.
6. Jeśli publikujesz pliki binarne, dodaj krótką instrukcję użycia.
7. Przed publikacją sprawdź, czy tag wskazuje właściwy commit.

---

## 11. Najczęstsze błędy

- utworzenie release bez wcześniejszego sprawdzenia tagu,
- brak opisu zmian,
- wrzucanie niepodpisanych lub nieopisanych plików,
- mieszanie wersji testowych i stabilnych,
- publikowanie artefaktów, których nie da się odtworzyć.

---

## Powiązane artykuły

- [Git i GitHub](#wiki-git-github)
- [GitHub Releases i GitHub Packages](#wiki-github_releases_packages)
- [Praca w zespole inżynierskim](#wiki-praca-w-zespole)
- [Tworzenie nowego repozytorium zadaniowego](#wiki-zakladanie-repozytorium-zadaniowego)

## Zasoby

- GitHub Docs: About releases
- GitHub CLI manual
- Semantic Versioning: https://semver.org/

---
*Ostatnia aktualizacja: 2026-03-20*
*Autor: Codex / Zespół Laboratorium*
