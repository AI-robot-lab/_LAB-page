# Praca w zespole inżynierskim

Skuteczna praca w laboratorium robotycznym wymaga nie tylko wiedzy technicznej, ale też wspólnych zasad komunikacji, dokumentowania decyzji i organizacji zadań. W projektach łączących oprogramowanie, sprzęt, AI i testy na robocie największe opóźnienia zwykle nie wynikają z trudności algorytmicznych, lecz z braku synchronizacji między członkami zespołu.

---

## 1. Dlaczego to ważne?

W projektach inżynierskich kilka osób pracuje równolegle nad:

- kodem źródłowym,
- konfiguracją środowisk,
- integracją czujników i aktuatorów,
- eksperymentami na robocie,
- dokumentacją i raportami.

Bez ustalonego sposobu pracy łatwo o sytuacje, w których:

- dwie osoby modyfikują ten sam moduł bez wiedzy o sobie,
- ktoś testuje robota na nieaktualnej wersji kodu,
- wyniki eksperymentu nie są powtarzalne,
- decyzje projektowe giną w prywatnych wiadomościach,
- nowy członek zespołu nie wie, od czego zacząć.

---

## 2. Podstawowe narzędzia pracy zespołowej

### 2.1. Git i GitHub

Git służy do wersjonowania kodu, a GitHub do współpracy nad repozytoriami, przeglądu zmian i śledzenia historii projektu.

Najważniejsze zastosowania:

- praca na branchach,
- pull requesty,
- code review,
- issues i planowanie prac,
- publikacja dokumentacji, release'ów i paczek.

> W naszym WIKI zagadnienia Git/GitHub znajdują się teraz w kategorii **Praca w zespole**.

### 2.2. Komunikator zespołowy

Zespół powinien używać jednego głównego kanału komunikacji synchronicznej, np.:

- Microsoft Teams,
- Slack,
- Discord,
- Mattermost.

Najlepiej rozdzielać kanały tematycznie, np.:

- `#ogloszenia`,
- `#percepcja`,
- `#kognicja`,
- `#interakcja`,
- `#robot-hardware`,
- `#help`.

### 2.3. Dokumentacja współdzielona

Dokumentacja powinna być dostępna dla całego zespołu i aktualizowana na bieżąco. Sprawdzają się:

- Markdown w repozytorium,
- WIKI projektu,
- Google Docs / Office 365 dla roboczych notatek,
- README i pliki `docs/` dla instrukcji technicznych.

### 2.4. Narzędzia do planowania zadań

Do planowania zadań przydatne są:

- GitHub Issues,
- GitHub Projects,
- tablice Kanban,
- checklisty w pull requestach,
- cotygodniowe plany pracy.

---

## 3. Zasady komunikacji technicznej

### 3.1. Komunikuj status jasno i krótko

Dobra aktualizacja statusu odpowiada na trzy pytania:

1. Co zrobiłem?
2. Co robię teraz?
3. Co mnie blokuje?

Przykład:

> Dziś uruchomiłem pipeline kamery RGB-D na Jetsonie, dodałem logowanie klatek do ROS2 bag. Teraz pracuję nad synchronizacją timestampów. Blokada: niestabilne połączenie SSH z robotem po zmianie sieci Wi‑Fi.

### 3.2. Ustalaj decyzje na piśmie

Ważne decyzje techniczne nie powinny istnieć tylko „w rozmowie”. Po ustaleniu warto zapisać:

- kontekst decyzji,
- wybraną opcję,
- odrzucone alternatywy,
- wpływ na inne moduły,
- osobę odpowiedzialną.

### 3.3. Nie ukrywaj problemów integracyjnych

W robotyce bardzo często problem leży „pomiędzy” modułami. Jeśli coś nie działa:

- zgłoś to wcześnie,
- dodaj logi, screeny lub komendy,
- opisz wersję kodu i środowiska,
- zaznacz, czy problem występuje na symulatorze, lokalnie czy na robocie.

---

## 4. Dobre praktyki pracy w zespole

### 4.1. Jedno zadanie = jeden branch

Każdą większą zmianę realizuj w osobnej gałęzi, np.:

```bash
git switch -c feature/lidar-time-sync
```

Dzięki temu:

- łatwiej zrobić review,
- łatwiej wycofać zmianę,
- łatwiej powiązać kod z zadaniem.

### 4.2. Małe, czytelne commity

Zamiast jednego dużego commita po kilku dniach pracy lepiej robić kilka mniejszych, logicznych kroków.

Dobre komunikaty commitów:

- `Add ROS2 node for camera diagnostics`
- `Fix timestamp conversion in lidar pipeline`
- `Update README with Jetson SSH workflow`

### 4.3. Pull request jako miejsce rozmowy technicznej

Pull request nie służy tylko do scalenia kodu. To miejsce, gdzie:

- opisujesz cel zmiany,
- wskazujesz wpływ na system,
- prosisz o review,
- dokumentujesz decyzje projektowe.

### 4.4. Testuj zanim poprosisz o review

Przed otwarciem PR warto sprawdzić:

- czy kod się uruchamia,
- czy testy przechodzą,
- czy README / dokumentacja są zaktualizowane,
- czy nie dodałeś sekretów, dużych plików lub artefaktów tymczasowych.

---

## 5. Organizacja pracy na robocie

Robot jest zasobem współdzielonym, więc potrzebne są dodatkowe zasady.

### 5.1. Rezerwacja czasu pracy

Jeśli testy wymagają fizycznego dostępu do robota, ustal wcześniej:

- okno czasowe,
- cel testu,
- osobę prowadzącą,
- plan awaryjny.

### 5.2. Zawsze zapisuj stan testu

Po sesji testowej zanotuj:

- branch / commit,
- parametry eksperymentu,
- wersję modelu,
- użyte sensory,
- wynik i obserwacje,
- ewentualne problemy bezpieczeństwa.

### 5.3. Najpierw bezpieczeństwo

Przed testem na robocie upewnij się, że:

- przestrzeń robocza jest wolna,
- znasz procedurę zatrzymania awaryjnego,
- ruch robota jest ograniczony do potrzebnego zakresu,
- ktoś wie, że prowadzisz test.

---

## 6. Onboarding nowego członka zespołu

Nowa osoba powinna już pierwszego dnia otrzymać:

- link do organizacji GitHub i repozytoriów,
- opis aktualnych zadań,
- dostęp do komunikatora,
- instrukcję uruchomienia środowiska,
- instrukcję dostępu do robota przez SSH,
- wskazanie opiekuna technicznego.

Polecana ścieżka startowa:

1. przeczytaj README projektu,
2. skonfiguruj Git i klucze SSH,
3. sklonuj repozytorium,
4. uruchom projekt lokalnie,
5. wykonaj małą zmianę dokumentacyjną,
6. otwórz pierwszy pull request.

---

## 7. Minimalny standard współpracy w laboratorium

Każdy członek zespołu powinien:

- pracować na branchach,
- robić commity z opisem,
- aktualizować dokumentację,
- zgłaszać blokery wcześnie,
- nie nadpisywać cudzej pracy bez uzgodnienia,
- dbać o powtarzalność eksperymentów,
- zostawiać po sobie czytelny stan projektu.

---

## Powiązane artykuły

- [Git i GitHub](#wiki-git-github)
- [GitHub Releases i Packages](#wiki-github_releases_packages)
- [Bash i SSH do robota (Jetson)](#wiki-bash-ssh-jetson)
- [Tworzenie nowego repozytorium zadaniowego](#wiki-zakladanie-repozytorium-zadaniowego)

## Zasoby

- GitHub Docs: praca z pull requestami i repository permissions
- Dokumentacja zespołowa projektu
- README repozytoriów zespołowych

---
*Ostatnia aktualizacja: 2026-03-20*
*Autor: Codex / Zespół Laboratorium*
