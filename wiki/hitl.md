# Hardware-in-the-Loop (HIL) oraz Human-in-the-Loop (HuIL)

## Wprowadzenie

**Hardware-in-the-Loop (HIL)** i **Human-in-the-Loop (HuIL)** to dwa komplementarne paradygmaty projektowania, testowania i walidacji złożonych systemów cyber-fizycznych, w których łączy się modele obliczeniowe, rzeczywisty sprzęt, środowisko wykonawcze oraz udział człowieka. W robotyce, automatyce, systemach autonomicznych, lotnictwie, motoryzacji czy medycynie oba podejścia są stosowane po to, aby zmniejszyć ryzyko wdrożeniowe, poprawić bezpieczeństwo, zwiększyć wiarygodność wyników eksperymentów i przyspieszyć iteracyjne doskonalenie algorytmów.

W praktyce laboratoryjnej HIL odpowiada przede wszystkim na pytanie: **czy sterownik, sensory, aktuatory i magistrale komunikacyjne zachowują się poprawnie po połączeniu z realistycznym modelem reszty systemu?** Z kolei HuIL odpowiada na pytanie: **czy człowiek współpracujący z systemem, nadzorujący go lub będący odbiorcą jego działania, może efektywnie, bezpiecznie i przewidywalnie wpływać na przebieg zadania?**

W nowoczesnych systemach robotycznych oba podejścia coraz częściej występują razem. Robot humanoidalny może być testowany w pętli HIL z rzeczywistym sterownikiem kończyn i zasymulowaną dynamiką całego ciała, a jednocześnie w pętli HuIL z operatorem dostarczającym komendy głosowe, korekty trajektorii lub oceny jakości interakcji.

## Dlaczego te podejścia są ważne?

### Główne motywacje inżynierskie

1. **Redukcja ryzyka** – wczesne wykrywanie błędów integracyjnych i błędów interfejsów.
2. **Bezpieczeństwo eksperymentów** – możliwość testowania scenariuszy awaryjnych bez narażania ludzi i kosztownego sprzętu.
3. **Oszczędność czasu i kosztów** – mniej prób bezpośrednio na pełnym stanowisku lub robocie.
4. **Lepsza reprodukowalność** – możliwość odtwarzania identycznych warunków testowych.
5. **Walidacja człowieka jako elementu systemu** – uwzględnienie ograniczeń percepcyjnych, poznawczych i motorycznych operatora.
6. **Zgodność z praktyką przemysłową** – oba podejścia są standardem w certyfikacji i rozwoju systemów krytycznych.

### Znaczenie naukowe

Z perspektywy naukowej HIL i HuIL umożliwiają badanie systemu nie tylko jako zestawu algorytmów, ale jako **układu osadzonego w rzeczywistym kontekście fizycznym i społecznym**. Dzięki temu można wiarygodniej analizować:

- stabilność sterowania w warunkach opóźnień,
- wpływ nieliniowości i szumów sensorycznych,
- odporność na uszkodzenia częściowe,
- obciążenie poznawcze operatora,
- jakość współpracy człowiek–robot,
- kompromisy między autonomią a nadzorem człowieka.

## Hardware-in-the-Loop (HIL)

## Definicja i idea

**Hardware-in-the-Loop** to metoda testowania, w której część systemu występuje w postaci rzeczywistego sprzętu, a pozostała część jest symulowana w czasie rzeczywistym. Najczęściej rzeczywistym elementem jest sterownik, komputer pokładowy, ECU, moduł sensoryczny, napęd albo układ komunikacyjny, natomiast obiekt sterowania lub otoczenie reprezentowane są przez model numeryczny.

Kluczową cechą HIL jest to, że sprzęt komunikuje się z symulatorem tak, jakby współpracował z rzeczywistym obiektem. Oznacza to, że testowana jednostka „widzi” realistyczne sygnały wejściowe i generuje sygnały wyjściowe, które są natychmiast uwzględniane przez model.

### Typowy schemat HIL

- **testowany sprzęt**: sterownik, komputer czasu rzeczywistego, mikrokontroler, PLC, napęd,
- **symulator czasu rzeczywistego**: model dynamiki robota, pojazdu lub procesu,
- **interfejs I/O**: ADC/DAC, PWM, CAN, EtherCAT, UART, GPIO, ROS 2 bridge,
- **warstwa monitoringu**: logowanie, synchronizacja czasu, analiza błędów,
- **warstwa scenariuszy testowych**: zakłócenia, awarie, zmiany parametrów.

## Poziomy realizacji HIL

### 1. Processor-in-the-Loop (PIL)

Na tym poziomie rzeczywisty procesor wykonuje kod docelowy, lecz otoczenie pozostaje symulowane. Jest to krok pośredni między pełną symulacją a klasycznym HIL.

### 2. Controller-in-the-Loop

Testowany jest rzeczywisty sterownik z pełnymi wejściami i wyjściami, podczas gdy model obiektu działa w symulatorze czasu rzeczywistego.

### 3. Power Hardware-in-the-Loop (PHIL)

Odmiana HIL dla układów energoelektronicznych i napędowych, w której w pętli występuje także rzeczywista wymiana mocy. Wymaga to wzmacniaczy mocy, bardzo ostrożnej kompensacji opóźnień i rygorystycznej analizy stabilności.

### 4. Network-in-the-Loop

Nacisk położony jest na testowanie sieci komunikacyjnych, opóźnień, jittera, utraty pakietów i odporności systemu rozproszonego.

## Wymagania techniczne HIL

### Czas rzeczywisty

Najważniejszym wymaganiem jest zachowanie deterministycznego kroku symulacji. Jeśli model dynamiki aktualizuje stan co 1 ms, a sterownik oczekuje odpowiedzi w takim samym okresie, cała pętla musi działać w przewidywalnym budżecie czasowym.

Problemy pojawiające się przy naruszeniu czasu rzeczywistego:

- niestabilność pętli sterowania,
- błędna ocena działania regulatora,
- artefakty wynikające z aliasingu i opóźnień,
- niepoprawne odwzorowanie czujników i napędów.

### Wierność modelu

HIL nie jest lepszy od modelu, który zasila testowany sprzęt. Model powinien obejmować:

- dynamikę mechaniczną,
- ograniczenia aktorów,
- opóźnienia i nasycenia,
- charakterystyki sensorów,
- szumy i dryft,
- kolizje, tarcie i podatność,
- zjawiska termiczne lub energetyczne, jeśli wpływają na sterowanie.

### Synchronizacja i znaczniki czasu

W środowiskach rozproszonych konieczna jest spójna synchronizacja czasu. Bez niej trudno odróżnić faktyczny błąd regulatora od błędu asynchronicznego próbkowania.

## Architektura HIL w robotyce

### Przykład dla robota mobilnego

1. Komputer sterujący uruchamia planowanie ruchu i regulator prędkości.
2. Symulator czasu rzeczywistego emuluje dynamikę platformy i kontakt kół z podłożem.
3. Wirtualne czujniki generują dane IMU, enkoderów, LiDAR-u lub kamery.
4. Sterownik odbiera dane tak, jakby pochodziły z rzeczywistego robota.
5. Komendy sterujące wracają do modelu i wpływają na stan systemu.

### Przykład dla robota humanoidalnego

W przypadku humanoida HIL może obejmować:

- rzeczywiste sterowniki stawów,
- zasymulowaną dynamikę całego ciała,
- model kontaktu stopy z podłożem,
- emulację sił reakcji podłoża,
- wymuszenia związane z zaburzeniami równowagi,
- zewnętrzny system nadzorczy bezpieczeństwa.

Takie podejście pozwala sprawdzić np. czy regulator postawy, chodu lub manipulacji zachowa stabilność przy realistycznych opóźnieniach magistrali i ograniczeniach momentu napędów.

## Zastosowania HIL

### Motoryzacja

- testy ECU i ADAS,
- walidacja systemów hamowania i steer-by-wire,
- badanie odporności na awarie sensorów,
- testy algorytmów zarządzania energią.

### Lotnictwo i kosmonautyka

- testy autopilotów,
- integracja sensorów nawigacyjnych,
- walidacja systemów kontroli lotu,
- symulacja awarii i scenariuszy granicznych.

### Robotyka przemysłowa i usługowa

- uruchamianie sterowników manipulatorów,
- badanie bezpieczeństwa współpracy z człowiekiem,
- testy planowania trajektorii i reakcji na przeszkody,
- integracja percepcji z kontrolą ruchu.

### Energetyka i systemy cyber-fizyczne

- sterowanie przekształtnikami,
- stabilizacja mikro-sieci,
- testy zabezpieczeń i układów nadzorczych.

## Zalety HIL

- wysoka wiarygodność testów integracyjnych,
- możliwość badania awarii trudnych lub niebezpiecznych do odtworzenia fizycznie,
- wcześniejsze wykrywanie problemów sprzętowo-programowych,
- krótszy cykl rozwoju,
- łatwiejsze strojenie regulatorów,
- dobre środowisko do walidacji bezpieczeństwa funkcjonalnego.

## Ograniczenia HIL

- kosztowna infrastruktura czasu rzeczywistego,
- konieczność budowy i utrzymania dokładnych modeli,
- ryzyko fałszywego poczucia bezpieczeństwa przy zbyt uproszczonych modelach,
- trudności z emulacją zjawisk rzadkich lub słabo poznanych,
- złożona integracja interfejsów sprzętowych.

## Human-in-the-Loop (HuIL)

## Definicja i idea

**Human-in-the-Loop** oznacza takie projektowanie lub testowanie systemu, w którym człowiek stanowi aktywny element pętli decyzyjnej, sterującej, oceniającej albo walidacyjnej. Człowiek może pełnić rolę operatora, użytkownika końcowego, eksperta dziedzinowego, teleoperatora, recenzenta wyników modelu lub źródła etykiet dla danych uczących.

W odróżnieniu od HIL, gdzie „prawdziwy” jest przede wszystkim komponent fizyczny, w HuIL centralne znaczenie ma **rzeczywisty udział człowieka wraz z jego percepcją, uwagą, błędami, heurystykami i ograniczeniami poznawczymi**.

## Role człowieka w pętli

### 1. Człowiek jako operator

Steruje systemem bezpośrednio lub półautonomicznie, np. przez joystick, gest, mowę, panel operatorski lub interfejs VR.

### 2. Człowiek jako nadzorca

Monitoruje działanie autonomicznego systemu i przejmuje kontrolę w sytuacjach niepewnych, niebezpiecznych lub nieobjętych modelem.

### 3. Człowiek jako źródło informacji semantycznej

Dostarcza kontekst, cel zadania, priorytety lub ograniczenia, których system nie umie wydedukować z samych danych sensorycznych.

### 4. Człowiek jako ewaluator

Ocenia jakość działania systemu: użyteczność, zaufanie, komfort, akceptowalność społeczną, poziom obciążenia i intuicyjność interakcji.

### 5. Człowiek jako nauczyciel systemu

Dostarcza demonstracji, korekt, etykiet lub informacji zwrotnej wykorzystywanej w uczeniu nadzorowanym, aktywnym lub przez wzmacnianie z ludzką preferencją.

## Poziomy autonomii a HuIL

Człowiek w pętli może uczestniczyć na różnych poziomach autonomii:

- **manual control** – człowiek steruje każdym krokiem,
- **shared control** – sterowanie dzielone między człowieka i automat,
- **supervisory control** – człowiek zleca cel, system realizuje szczegóły,
- **human-on-the-loop** – system działa samodzielnie, lecz człowiek nadzoruje i może interweniować,
- **human-out-of-the-loop** – brak bieżącego udziału człowieka; ten wariant jest istotny jako punkt porównania, a nie przykład HuIL.

W praktyce projektowej kluczowe jest świadome ustalenie, **na którym etapie decyzji** człowiek ma realny wpływ i ile czasu ma na reakcję.

## Aspekty naukowe HuIL

### Ergonomia poznawcza

HuIL wymaga uwzględnienia ograniczeń człowieka, takich jak:

- ograniczona pojemność uwagi,
- zmęczenie i spadek czujności,
- opóźnienia reakcji,
- podatność na przeciążenie informacyjne,
- błędy interpretacyjne wynikające z niejasnego interfejsu.

### Czynniki psychologiczne

Na skuteczność pętli człowiek–system wpływają:

- zaufanie do automatyki,
- poziom wyjaśnialności działania algorytmu,
- poczucie sprawczości,
- stres i presja czasu,
- przewidywalność zachowania robota.

### Mierniki badawcze

Najczęściej stosowane wskaźniki obejmują:

- czas reakcji operatora,
- skuteczność wykonania zadania,
- liczbę interwencji człowieka,
- częstość błędów,
- obciążenie poznawcze, np. NASA-TLX,
- wskaźniki zaufania i akceptacji,
- bezpieczeństwo interakcji,
- jakość doświadczenia użytkownika.

## Typowe zastosowania HuIL

### Teleoperacja i robotyka zdalna

Operator steruje robotem w środowiskach niebezpiecznych, trudno dostępnych albo słabo poznanych: w ratownictwie, inspekcji przemysłowej, medycynie czy eksploracji.

### Systemy wspomagania decyzji

Algorytm proponuje rozwiązania, lecz ostateczna decyzja należy do człowieka, np. w diagnostyce, bezpieczeństwie, planowaniu misji lub zarządzaniu ruchem.

### Uczenie maszynowe z udziałem człowieka

- aktywne wybieranie próbek do etykietowania,
- uczenie z demonstracji,
- reinforcement learning from human feedback,
- korekta trajektorii i polityk sterowania na podstawie preferencji eksperta.

### Roboty społeczne i asystujące

HuIL jest tu nieodzowny, ponieważ efektywność systemu zależy od jakości interakcji, akceptacji użytkownika i dopasowania zachowania robota do norm społecznych.

## Projektowanie systemów Human-in-the-Loop

### Kluczowe zasady

1. **Człowiek nie może być traktowany jak idealny sensor lub idealny regulator.**
2. **Interfejs musi wspierać świadomość sytuacyjną.**
3. **Automatyzacja powinna być czytelna i wyjaśnialna.**
4. **Zakres odpowiedzialności człowieka i systemu musi być jednoznaczny.**
5. **Mechanizmy przejęcia i oddania kontroli muszą być zaprojektowane jawnie.**

### Zagrożenia projektowe

- ironia automatyzacji: im lepsza automatyka, tym trudniej człowiekowi skutecznie interweniować po długim okresie bezczynności,
- przeuczenie operatora do scenariuszy laboratoryjnych,
- zbyt duża liczba alarmów prowadząca do alarm fatigue,
- brak transparentności modelu decyzyjnego,
- mylenie „nadzoru” z realną możliwością interwencji.

## HIL a HuIL — różnice i podobieństwa

## Główna różnica

- **HIL** koncentruje się na włączeniu rzeczywistego komponentu sprzętowego do pętli testowej.
- **HuIL** koncentruje się na włączeniu rzeczywistego człowieka do pętli decyzyjnej lub badawczej.

## Wspólny cel

Oba podejścia zmierzają do tego, aby testowany system był oceniany w warunkach bliższych rzeczywistości niż w czystej symulacji.

## Tabela porównawcza

| Cecha | Hardware-in-the-Loop | Human-in-the-Loop |
|---|---|---|
| Główny obiekt realizmu | Sprzęt i interfejsy fizyczne | Zachowanie, decyzje i ograniczenia człowieka |
| Typowe pytanie badawcze | Czy sterownik działa poprawnie z realistycznym obiektem? | Czy człowiek potrafi bezpiecznie i efektywnie współpracować z systemem? |
| Kluczowe ryzyko | Niedokładny model lub niedeterministyczny czas | Błędy człowieka, przeciążenie, niejasny interfejs |
| Mierniki | opóźnienie, stabilność, błąd regulacji, integralność sygnałów | czas reakcji, użyteczność, zaufanie, liczba interwencji |
| Narzędzia | symulator RT, I/O, magistrale, analizatory | interfejs HMI, ankiety, eye-tracking, logi operatora |
| Główne zastosowania | sterowanie, integracja, walidacja bezpieczeństwa | teleoperacja, nadzór, uczenie z człowiekiem, HRI |

## Połączenie obu podejść

Najbardziej wartościowe eksperymenty często łączą HIL i HuIL. Przykładowo:

- sterownik robota działa na rzeczywistym komputerze i napędach w pętli HIL,
- dynamika otoczenia jest symulowana w czasie rzeczywistym,
- operator wydaje polecenia przez interfejs głosowy lub konsolę,
- człowiek ocenia zachowanie systemu i interweniuje przy niepewności.

Takie środowisko jest szczególnie cenne przy badaniu robotów humanoidalnych, systemów współdzielonego sterowania, asystentów mobilnych i stanowisk rehabilitacyjnych.

## Metodologia eksperymentu HIL/HuIL

### 1. Definicja celu badania

Należy jasno określić, czy celem jest:

- walidacja regulatora,
- test odporności na awarie,
- analiza interakcji człowiek–robot,
- porównanie poziomów autonomii,
- ocena jakości interfejsu,
- zebranie danych do uczenia modeli.

### 2. Dobór poziomu realizmu

Nie każdy eksperyment wymaga pełnego realizmu fizycznego i udziału ekspertów dziedzinowych. Koszt i złożoność powinny być proporcjonalne do pytania badawczego.

### 3. Identyfikacja zmiennych niezależnych i zależnych

Przykłady:

- poziom opóźnienia w komunikacji,
- dokładność modelu sensora,
- stopień autonomii systemu,
- rodzaj interfejsu operatora,
- doświadczenie uczestnika,
- czas wykonania zadania,
- liczba błędów i interwencji.

### 4. Projekt scenariuszy testowych

Scenariusze powinny obejmować zarówno przypadki nominalne, jak i graniczne:

- zakłócenia sensoryczne,
- utratę komunikacji,
- błędne wykrycia percepcyjne,
- konflikt poleceń człowieka i automatyki,
- awarie częściowe aktuatorów,
- niepewne warunki otoczenia.

### 5. Rejestracja i analiza danych

Należy logować równocześnie:

- stany modelu,
- sygnały sterujące,
- znaczniki czasu,
- zdarzenia interfejsu użytkownika,
- działania operatora,
- metryki bezpieczeństwa i wydajności.

### 6. Walidacja statystyczna i jakościowa

W HuIL same logi systemowe często nie wystarczą. Potrzebne mogą być również:

- kwestionariusze,
- wywiady po zadaniu,
- analiza nagrań wideo,
- analiza błędów człowieka,
- porównanie grup uczestników.

## Przykład architektury badawczej dla laboratorium robotów

### Scenariusz: robot humanoidalny pomagający człowiekowi

Załóżmy, że badamy robota humanoidalnego wykonującego zadanie podania przedmiotu operatorowi.

**Warstwa HIL:**
- rzeczywisty kontroler ramienia,
- rzeczywiste sterowniki chwytaka,
- emulowana dynamika pełnego ciała i kontaktów,
- emulowane opóźnienia czujników wizyjnych.

**Warstwa HuIL:**
- człowiek wydaje polecenie głosem,
- operator może skorygować chwyt lub pozycję celu,
- uczestnik ocenia naturalność ruchu i poczucie bezpieczeństwa,
- badacz mierzy czas wykonania, liczbę korekt i minimalny dystans bezpieczeństwa.

Taki eksperyment pozwala jednocześnie zbadać:

- poprawność sterowania,
- odporność na opóźnienia,
- jakość interfejsu człowiek–robot,
- subiektywne zaufanie użytkownika,
- wpływ poziomu autonomii na efektywność zadania.

## Dobre praktyki

### Dla HIL

- zaczynaj od modeli prostych, ale walidowanych eksperymentalnie,
- dokumentuj wszystkie założenia modelu i ich zakres obowiązywania,
- kontroluj jitter, opóźnienia i rozdzielczość próbkowania,
- oddzielaj błędy modelu od błędów implementacji,
- testuj scenariusze awaryjne w sposób systematyczny.

### Dla HuIL

- projektuj interfejs pod kątem czytelności i przewidywalności,
- nie przeciążaj operatora nadmiarem informacji,
- uwzględniaj szkolenie uczestników i efekt uczenia,
- analizuj nie tylko średnią skuteczność, ale też rozkład błędów,
- stosuj zarówno metryki obiektywne, jak i subiektywne.

### Dla połączonych eksperymentów HIL/HuIL

- synchronizuj wszystkie źródła danych wspólnym zegarem,
- definiuj precyzyjnie momenty przekazania kontroli,
- uwzględniaj fail-safe i emergency stop,
- dbaj o etykę badań z udziałem ludzi,
- utrzymuj powtarzalność scenariuszy mimo udziału operatorów.

## Najczęstsze błędy interpretacyjne

1. **„Jeśli działa w HIL, to na pewno zadziała w rzeczywistości.”** – nie, bo nadal istnieją niezamodelowane zjawiska.
2. **„Skoro człowiek nadzoruje system, to system jest bezpieczny.”** – nie, jeśli interwencja jest spóźniona lub nieintuicyjna.
3. **„Wystarczy dodać operatora i to już jest HuIL.”** – nie, jeśli rola człowieka nie ma mierzalnego wpływu na pętlę.
4. **„Realizm maksymalny zawsze jest najlepszy.”** – nie, bo może nieproporcjonalnie zwiększać koszt bez wzrostu wartości poznawczej.

## Powiązanie z innymi obszarami laboratorium

Podejścia HIL i HuIL silnie łączą się z następującymi tematami:

- [Interakcja Człowiek-Robot](#wiki-hri)
- [Bezpieczeństwo robotów](#wiki-safety)
- [Framework PCA](#wiki-pca-framework)
- [Transfer Sim-to-Real](#wiki-sim-to-real)
- [Teoria sterowania](#wiki-control-theory)
- [Fuzja sensoryczna](#wiki-sensor-fusion)
- [Planowanie ruchu](#wiki-motion-planning)

## Podsumowanie

Hardware-in-the-Loop i Human-in-the-Loop są fundamentalnymi metodami budowy wiarygodnych systemów robotycznych oraz cyber-fizycznych. HIL wzmacnia realizm techniczny przez integrację rzeczywistego sprzętu z symulacją czasu rzeczywistego, a HuIL wzmacnia realizm operacyjny i poznawczy przez włączenie człowieka jako aktywnego elementu systemu. Wspólnie tworzą one pomost między czystą symulacją a wdrożeniem w świecie rzeczywistym.

Dla laboratoriów robotyki oznacza to możliwość prowadzenia badań, które są jednocześnie bezpieczne, powtarzalne, naukowo rzetelne i praktycznie użyteczne. W szczególności przy robotach humanoidalnych, systemach współdzielonej kontroli i aplikacjach HRI połączenie HIL z HuIL staje się nie tyle dodatkiem, ile centralnym elementem procesu badawczo-rozwojowego.

## Zasoby i słowa kluczowe

### Słowa kluczowe

- cyber-physical systems
- real-time simulation
- hardware-in-the-loop
- power hardware-in-the-loop
- human-in-the-loop
- shared autonomy
- supervisory control
- human-robot interaction
- explainable autonomy
- safety validation

### Polecane kierunki dalszego studiowania

- standardy bezpieczeństwa funkcjonalnego,
- modele człowieka w pętli sterowania,
- teleoperacja z opóźnieniem,
- cyfrowe bliźniaki w robotyce,
- walidacja systemów autonomicznych,
- metodologia eksperymentów z udziałem człowieka.

---
*Ostatnia aktualizacja: 2026-03-20*
*Autor: OpenAI Codex*
