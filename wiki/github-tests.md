# Testy w GitHub — profesjonalny przewodnik dla studentów

Testy uruchamiane w ekosystemie GitHub to dziś jeden z podstawowych elementów profesjonalnej pracy zespołowej nad oprogramowaniem. Pozwalają automatycznie sprawdzić, czy kod działa poprawnie, spełnia wymagania jakościowe i nie psuje wcześniej działających funkcji. Dla studentów jest to szczególnie ważne, ponieważ uczy pracy zgodnej ze standardami stosowanymi w firmach technologicznych, laboratoriach badawczych i projektach open source.

Najczęściej mówiąc o „testach w GitHub”, mamy na myśli połączenie repozytorium z mechanizmami automatyzacji, głównie **GitHub Actions**, które uruchamiają testy po `push`, przy otwarciu `pull requesta`, przed wdrożeniem albo według harmonogramu.

---

## 1. Czym są testy w GitHub?

**Testy w GitHub** to zautomatyzowane sprawdzenia wykonywane na kodzie przechowywanym w repozytorium. Ich celem jest szybkie wykrycie błędów, niezgodności i regresji bez konieczności ręcznego uruchamiania wszystkich kontroli przez człowieka.

W praktyce testy mogą obejmować:

- **testy jednostkowe** — sprawdzają pojedyncze funkcje, klasy lub moduły,
- **testy integracyjne** — sprawdzają współpracę kilku elementów systemu,
- **testy end-to-end** — weryfikują pełne scenariusze użytkownika,
- **linting i formatowanie** — kontrola jakości stylu kodu,
- **testy bezpieczeństwa** — wykrywanie znanych podatności i sekretów,
- **budowanie projektu** — sprawdzenie, czy aplikacja daje się zbudować i uruchomić.

Na GitHubie wyniki takich kontroli są widoczne bezpośrednio przy commicie i pull requeście. Dzięki temu zespół od razu widzi, czy zmiana jest bezpieczna do dalszego review lub scalenia.

---

## 2. Po co wykonuje się testy?

### 2.1. Aby szybciej wykrywać błędy

Im wcześniej znajdziesz błąd, tym taniej i łatwiej go naprawić. Jeśli problem zostanie wykryty już po wysłaniu zmian do repozytorium, autor może od razu poprawić kod, zamiast dowiedzieć się o awarii dopiero podczas prezentacji projektu, laboratorium albo wdrożenia.

### 2.2. Aby zapobiegać regresjom

Regresja oznacza sytuację, w której nowa zmiana psuje coś, co wcześniej działało poprawnie. Dobrze zaprojektowany zestaw testów działa jak siatka bezpieczeństwa: pozwala rozwijać projekt bez ciągłego lęku, że przypadkowo usuniemy poprawne zachowanie systemu.

### 2.3. Aby ujednolicić jakość pracy zespołu

W projektach studenckich i badawczych kilka osób często pracuje równolegle nad różnymi modułami. Automatyczne testy wymuszają wspólny standard jakości niezależnie od doświadczenia autorów.

### 2.4. Aby przyspieszyć code review

Recenzent nie powinien tracić czasu na sprawdzanie oczywistych rzeczy, takich jak brakujące zależności, błędy składni czy nieprzechodzące testy jednostkowe. Gdy podstawowe kontrole wykonuje pipeline, człowiek może skupić się na architekturze, czytelności i sensowności rozwiązania.

### 2.5. Aby budować wiarygodność projektu

Repozytorium, w którym testy są regularnie uruchamiane i przechodzą, wygląda profesjonalnie. To ważne zarówno w pracy zespołowej, jak i przy budowaniu portfolio studenta.

---

## 3. Jak działa proces testowania na GitHub?

Najczęstszy przebieg wygląda następująco:

1. Programista tworzy branch i wprowadza zmiany.
2. Wysyła commit lub otwiera pull request.
3. GitHub uruchamia workflow zdefiniowany w `.github/workflows/*.yml`.
4. Workflow instaluje zależności, buduje projekt i uruchamia testy.
5. Wyniki są publikowane przy commicie i pull requeście.
6. Jeśli testy przejdą, zmiana może zostać zrecenzowana i scalona.
7. Jeśli testy nie przejdą, autor analizuje logi, poprawia kod i wysyła kolejną wersję.

### 3.1. Kluczowe pojęcia

- **workflow** — pełny proces automatyzacji zapisany w pliku YAML,
- **event** — zdarzenie uruchamiające workflow, np. `push` lub `pull_request`,
- **job** — większy etap wykonywany na maszynie wirtualnej,
- **step** — pojedynczy krok w jobie, np. instalacja zależności,
- **runner** — środowisko, na którym wykonywane są zadania,
- **status checks** — wyniki kontroli widoczne w GitHub.

### 3.2. Najczęstsze momenty uruchamiania testów

Testy uruchamia się zwykle:

- po każdym `push` do brancha roboczego,
- przy każdym `pull request`,
- przed połączeniem z gałęzią `main`,
- po utworzeniu wersji release,
- cyklicznie, np. codziennie w nocy.

---

## 4. Jakie rodzaje testów warto stosować?

### 4.1. Testy jednostkowe

To fundament automatycznej kontroli jakości. Powinny być:

- szybkie,
- powtarzalne,
- niezależne od sieci i środowiska zewnętrznego,
- łatwe do uruchomienia lokalnie.

Przykład: sprawdzenie, czy funkcja obliczająca średnią zwraca poprawną wartość dla zadanych danych wejściowych.

### 4.2. Testy integracyjne

Służą do sprawdzenia współpracy kilku warstw systemu, np. API z bazą danych, modułu ROS2 z sensorem albo backendu z kolejką zadań.

### 4.3. Testy end-to-end

Odwzorowują realny scenariusz użytkownika. Są cenniejsze biznesowo, ale zwykle wolniejsze i bardziej podatne na niestabilność, dlatego należy stosować je rozsądnie.

### 4.4. Linting i formatowanie

Nie każdy błąd dotyczy logiki. Problemy stylu kodu, niespójne formatowanie lub nieużywane importy również obniżają jakość projektu. Narzędzia takie jak ESLint, Prettier, Ruff, Black czy Flake8 pomagają utrzymać spójny standard.

### 4.5. Kontrole bezpieczeństwa i zależności

W bardziej dojrzałych projektach warto dodać:

- skanowanie podatności,
- kontrolę licencji,
- wykrywanie wycieków sekretów,
- analizę zależności.

---

## 5. Przyjęte zasady dobrego testowania w GitHub

### 5.1. Testy muszą być uruchamialne lokalnie

Jeśli test działa tylko w chmurze, a nie działa na komputerze autora, jego naprawa będzie trudna. Dobra praktyka mówi: **najpierw uruchom test lokalnie, potem wysyłaj zmiany do GitHub**.

### 5.2. Pipeline powinien być szybki i czytelny

Zbyt wolne testy zniechęcają do ich używania. W projektach studenckich dobrym celem jest, aby podstawowy zestaw kontroli kończył się możliwie szybko, a pełniejsze testy można uruchamiać osobno.

### 5.3. Jedna awaria = jasna informacja o przyczynie

Logi powinny jednoznacznie wskazywać, co się nie udało. Lepiej mieć kilka logicznie rozdzielonych kroków niż jeden długi skrypt, po którym trudno znaleźć źródło problemu.

### 5.4. Nie ignoruj czerwonych statusów

Jeżeli workflow jest czerwony, nie należy scalać zmian „na szybko”, licząc, że później ktoś to naprawi. Taka praktyka bardzo szybko psuje kulturę jakości w zespole.

### 5.5. Chroń gałąź główną

Gałąź `main` lub `master` powinna być chroniona regułami, które wymagają przejścia testów przed scaleniem. Dzięki temu projekt zachowuje stabilny stan.

### 5.6. Testy powinny być deterministyczne

Ten sam kod powinien dawać ten sam wynik. Testy zależne od losowości, bieżącej godziny, internetu albo kolejności wykonania bez odpowiedniej kontroli prowadzą do tzw. flaky tests, czyli testów niestabilnych.

---

## 6. Jakich reguł należy przestrzegać?

Poniżej znajduje się zestaw reguł, które warto traktować jako minimalny standard pracy.

### 6.1. Reguły techniczne

- Każdy workflow trzymaj w repozytorium w katalogu `.github/workflows/`.
- Nazwy workflow i jobów powinny jasno opisywać ich cel.
- Wykorzystuj wersjonowane akcje, np. `actions/checkout@v4`, zamiast nieprecyzyjnych odwołań.
- Nie przechowuj haseł i tokenów bezpośrednio w plikach workflow.
- Używaj `secrets` i `variables` GitHub do danych wrażliwych.
- Rozdziel szybkie testy od ciężkich testów integracyjnych, jeśli projekt tego wymaga.
- Dbaj o spójność wersji środowiska, np. wersji Pythona lub Node.js.

### 6.2. Reguły organizacyjne

- Każdy pull request powinien przejść podstawowe testy przed review.
- Autor zmiany odpowiada za analizę logów i naprawę błędów.
- Nie wyłączaj testów tylko po to, aby „przepchnąć” zmianę.
- Jeśli test jest niestabilny, należy go naprawić albo tymczasowo odizolować z wyjaśnieniem.
- W opisie PR warto napisać, jakie testy uruchomiono lokalnie.

### 6.3. Reguły jakościowe

- Jeden test powinien sprawdzać jedną rzecz.
- Nazwy testów muszą opisywać zachowanie, a nie implementację.
- Testy powinny być proste do utrzymania.
- Każda poprawka błędu powinna — jeśli to możliwe — zawierać test odtwarzający problem.
- Dokumentacja projektu powinna zawierać instrukcję uruchamiania testów.

---

## 7. Typowe błędy początkujących

Studenci zaczynający pracę z testami w GitHub często popełniają podobne błędy:

- wrzucanie do pipeline tylko jednego polecenia bez lokalnej weryfikacji,
- brak pliku `requirements.txt`, `package-lock.json` albo innej kontroli zależności,
- testy zależne od konkretnej maszyny autora,
- brak rozróżnienia między błędem kodu a błędem konfiguracji workflow,
- łączenie wielu odpowiedzialności w jeden nieczytelny skrypt,
- traktowanie testów jako formalności zamiast narzędzia inżynierskiego.

Dobra praktyka polega na tym, aby myśleć o pipeline jak o **powtarzalnym eksperymencie**: ten sam proces ma dać ten sam wynik niezależnie od osoby uruchamiającej.

---

## 8. Praktyczny przykład krok po kroku

Poniżej znajduje się prosty przykład dla projektu w Pythonie. Załóżmy, że chcemy automatycznie uruchamiać testy jednostkowe przy każdym `push` i `pull request`.

### Krok 1. Przygotuj prosty projekt

Struktura katalogów może wyglądać tak:

```text
calculator-project/
├── calculator.py
├── test_calculator.py
└── requirements.txt
```

Plik `calculator.py`:

```python
def add(a, b):
    return a + b
```

Plik `test_calculator.py`:

```python
from calculator import add


def test_add_two_positive_numbers():
    assert add(2, 3) == 5


def test_add_negative_and_positive_number():
    assert add(-1, 4) == 3
```

Plik `requirements.txt`:

```text
pytest==8.3.5
```

### Krok 2. Sprawdź testy lokalnie

Najpierw uruchom:

```bash
python -m pip install -r requirements.txt
pytest
```

Jeśli testy nie działają lokalnie, nie ma sensu od razu wysyłać zmian do GitHub.

### Krok 3. Dodaj workflow GitHub Actions

Utwórz plik:

```text
.github/workflows/python-tests.yml
```

Wstaw do niego konfigurację:

```yaml
name: Python tests

on:
  push:
    branches:
      - main
      - develop
      - feature/**
  pull_request:
    branches:
      - main
      - develop

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: pytest
```

### Krok 4. Zacommituj i wyślij zmiany

```bash
git switch -c feature/python-tests
git add .
git commit -m "Add GitHub Actions tests for Python example"
git push origin feature/python-tests
```

### Krok 5. Otwórz pull request

Po otwarciu PR GitHub automatycznie uruchomi workflow. W zakładce:

- **Actions** zobaczysz przebieg wykonania,
- **Pull request** zobaczysz status checks,
- **Files changed** możesz prowadzić review kodu.

### Krok 6. Zinterpretuj wynik

- **zielony status** — testy przeszły,
- **czerwony status** — testy nie przeszły,
- **żółty status** — workflow jest w trakcie wykonywania,
- **szary status** — test nie został uruchomiony lub został pominięty.

### Krok 7. Dodaj regułę ochrony gałęzi

W ustawieniach repozytorium można ustawić, że branch `main` wymaga przejścia określonych status checks przed scaleniem. To jeden z najważniejszych mechanizmów jakościowych w pracy zespołowej.

### Krok 8. Rozwiń pipeline w kolejnym etapie

Gdy podstawowe testy już działają, można rozszerzyć workflow o:

- linting (`ruff`, `flake8`, `eslint`),
- sprawdzanie formatowania (`black`, `prettier`),
- macierz wersji środowiska,
- testy integracyjne,
- budowę artefaktów,
- automatyczne wdrożenie po spełnieniu warunków.

---

## 9. Przykładowy scenariusz akademicki

Załóżmy, że zespół studentów rozwija aplikację do analizy obrazu z kamery robota. Jedna osoba poprawia preprocessing obrazu, druga rozwija klasyfikator, a trzecia przygotowuje interfejs webowy.

Bez testów może dojść do sytuacji, w której:

- preprocessing zmienia format danych,
- klasyfikator przestaje działać,
- błąd zostaje zauważony dopiero podczas wspólnej demonstracji.

Z testami w GitHub:

- każda zmiana uruchamia kontrolę jakości,
- zespół szybciej wykrywa niezgodności interfejsów,
- łatwiej zachować stabilną gałąź główną,
- prowadzący projekt widzi bardziej profesjonalny proces pracy.

Właśnie dlatego testy nie są dodatkiem „na końcu”, lecz elementem codziennej inżynierii oprogramowania.

---

## 10. Podsumowanie

Testy w GitHub to połączenie **praktyki inżynierskiej**, **automatyzacji** i **odpowiedzialności zespołowej**. Dzięki nim można szybciej wykrywać błędy, ograniczać regresje, podnosić jakość kodu i budować stabilny proces wytwarzania oprogramowania.

Najważniejsze zasady, które warto zapamiętać:

- testuj lokalnie przed wysłaniem zmian,
- automatyzuj podstawowe kontrole w GitHub Actions,
- nie ignoruj błędów pipeline,
- chroń gałąź główną,
- utrzymuj testy szybkie, czytelne i deterministyczne,
- traktuj testy jako element kultury technicznej zespołu.

Dla studentów opanowanie tego procesu jest bardzo wartościowe, ponieważ odpowiada realnym standardom pracy stosowanym w branży IT, projektach naukowych i zespołach open source.

## Powiązane artykuły

- [Git i GitHub — kompleksowy poradnik](#wiki-git-github)
- [Praca w zespole inżynierskim](#wiki-praca-w-zespole)
- [GitHub Releases — czym są i jak je tworzyć](#wiki-github-releases)
- [Tworzenie nowego repozytorium zadaniowego](#wiki-zakladanie-repozytorium-zadaniowego)

## Zasoby

- GitHub Docs — GitHub Actions
- GitHub Docs — About status checks
- pytest Documentation
- Dokumentacja narzędzi lintujących używanych w projekcie

---
*Ostatnia aktualizacja: 2026-03-20*
*Autor: Codex / Zespół Laboratorium*
