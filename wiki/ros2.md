# ROS2 - Robot Operating System

## Wprowadzenie

**ROS2** (Robot Operating System 2) to otwarte oprogramowanie middleware do budowy aplikacji robotycznych. Jest to druga generacja frameworka ROS, zaprojektowana z myślą o systemach robotycznych działających w czasie rzeczywistym oraz środowiskach produkcyjnych.

### Dlaczego ROS2, a nie ROS1?

ROS1 (pierwsza generacja) był frameworkiem badawczym, który sprawdzał się w laboratoriach, lecz miał fundamentalne ograniczenia dla zastosowań produkcyjnych:

- **Centralny broker (roscore)** – w ROS1 wszystkie wiadomości przechodzą przez jeden proces `roscore`. Gdy on padnie, cały system przestaje działać. ROS2 eliminuje ten punkt awarii dzięki architekturze peer-to-peer opartej na DDS.
- **Brak wsparcia dla systemów real-time** – ROS1 nie gwarantował deterministycznych opóźnień komunikacji. ROS2 z DDS pozwala spełniać twarde wymagania czasowe (hard real-time).
- **Bezpieczeństwo** – ROS1 nie miał mechanizmów autentykacji ani szyfrowania. ROS2 pozwala na konfigurację SROS2 (Secure ROS2) z certyfikatami TLS.
- **Obsługa wielu platform** – ROS2 działa natywnie na Linux, Windows, macOS i systemach RTOS (np. FreeRTOS na mikrokontrolerach).

## Architektura ROS2

### DDS Middleware

ROS2 bazuje na standardzie **DDS** (Data Distribution Service), który zapewnia:
- Komunikację peer-to-peer bez centralnego brokera
- Quality of Service (QoS) profiles
- Automatyczne wykrywanie (discovery) węzłów w sieci
- Real-time performance

**Dlaczego DDS?** DDS to sprawdzony przemysłowy standard komunikacji (używany m.in. w systemach awionicznych, wojskowych, medycznych). Zamiast pisać własny protokół komunikacyjny od zera, twórcy ROS2 wybrali DDS, bo:
1. Jest otwarty i dobrze zdefiniowany (standard OMG)
2. Istnieje wiele implementacji (FastDDS, CycloneDDS, RTI Connext)
3. Implementuje mechanizmy QoS na poziomie warstwy transportowej

### Główne Komponenty

1. **Nodes (Węzły)** – podstawowe jednostki wykonawcze. Każdy węzeł to oddzielny proces (lub wątek), odpowiedzialny za jedną konkretną funkcję (np. czytanie kamery, planowanie trajektorii, sterowanie silnikiem). Rozbicie systemu na małe węzły to zasada **separation of concerns** – każdy węzeł jest łatwy do testowania i wymiany.
2. **Topics** – kanały komunikacji asynchronicznej. Publisher wysyła wiadomości na temat, Subscriber je odbiera. Nie muszą wiedzieć o sobie nawzajem – to **decoupling** (rozluźnianie zależności).
3. **Services** – komunikacja synchroniczna request-response. Klient wysyła żądanie i czeka na odpowiedź. Używane gdy potrzebujemy natychmiastowego wyniku (np. zapytanie o stan sensora).
4. **Actions** – długotrwałe zadania z możliwością śledzenia postępu i anulowania. Idealne dla zadań trwających sekundy lub minuty (np. przemieszczenie robota do celu).
5. **Parameters** – dynamiczna konfiguracja węzła podczas działania. Zamiast restartować węzeł po zmianie parametru, można go zmienić w locie.

### Schemat komunikacji

```
┌─────────────┐    topic /image    ┌──────────────────┐
│  Kamera     │ ─────────────────► │  Detektor obiektów│
│  (Publisher)│                    │  (Subscriber)    │
└─────────────┘                    └──────────────────┘

┌─────────────┐  service /detect   ┌──────────────────┐
│  Klient     │ ◄────────────────► │  Serwer detekcji │
│  (Request)  │                    │  (Response)      │
└─────────────┘                    └──────────────────┘

┌─────────────┐  action /navigate  ┌──────────────────┐
│  Navigator  │ ◄── feedback ─────►│  Action Server   │
│  (Client)   │ ──── goal ────────►│  (nawigacja)     │
└─────────────┘ ◄── result ────────└──────────────────┘
```

## Instalacja ROS2 Humble

ROS2 Humble Hawksbill to wersja LTS (Long Term Support) wspierana do maja 2027 roku. Wybieramy tę wersję dla stabilności – w środowisku produkcyjnym/laboratoryjnym ważna jest długoterminowa dostępność poprawek bezpieczeństwa.

```bash
# Ubuntu 22.04 LTS
# Krok 1: Ustawiamy locale – ROS2 wymaga UTF-8 dla poprawnego działania
# niektórych narzędzi (np. parsowanie nazw z polskimi znakami).
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Krok 2: Dodaj repozytorium ROS2
# 'universe' to niestandardowe repozytorium Ubuntu zawierające pakiety
# utrzymywane przez społeczność – wymagane dla niektórych zależności ROS2.
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y

# Pobieramy klucz GPG repozytorium ROS2.
# Klucz GPG pozwala apt zweryfikować, że pobierane pakiety naprawdę
# pochodzą od twórców ROS i nie zostały podmienione (bezpieczeństwo łańcucha dostaw).
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

# Krok 3: Dodaj źródło pakietów
# $(dpkg --print-architecture) → automatycznie wykrywa architekturę (amd64, arm64)
# $(. /etc/os-release && echo $UBUNTU_CODENAME) → wykrywa wersję Ubuntu (jammy)
# Dzięki temu ten sam skrypt działa na różnych maszynach.
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Krok 4: Instalacja
# ros-humble-desktop zawiera: rdzeń ROS2, RViz2, rqt, przykładowe pakiety.
# Alternatywnie ros-humble-ros-base (bez GUI) – na robotach bez monitora.
sudo apt update
sudo apt install ros-humble-desktop

# Krok 5: Skonfiguruj środowisko (dodaj do ~/.bashrc żeby działało automatycznie)
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Tworzenie Pakietu ROS2

Zanim napiszemy pierwszy węzeł, musimy stworzyć **pakiet** (package). Pakiet to podstawowa jednostka organizacji kodu w ROS2 – podobnie jak moduł w Python lub biblioteka w C++.

```bash
# Tworzymy przestrzeń roboczą (workspace)
# Konwencja: nazwa kończy się na _ws
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Tworzymy pakiet Python
# --build-type ament_python → używamy systemu budowania ament dla Pythona
# --dependencies rclpy std_msgs → automatycznie dodaje zależności do package.xml
ros2 pkg create --build-type ament_python --dependencies rclpy std_msgs my_robot_pkg

# Struktura pakietu po utworzeniu:
# my_robot_pkg/
# ├── my_robot_pkg/          ← katalog z kodem Python (musi mieć tę samą nazwę co pakiet)
# │   └── __init__.py        ← plik wymagany przez Python, może być pusty
# ├── package.xml            ← metadane: nazwa, wersja, zależności
# ├── setup.py               ← konfiguracja instalacji (entry points dla węzłów)
# └── setup.cfg              ← dodatkowa konfiguracja

# Budowanie workspace
cd ~/ros2_ws
colcon build --symlink-install
# --symlink-install → zamiast kopiować pliki Python, tworzy dowiązania symboliczne.
# Dzięki temu edycja pliku źródłowego od razu widoczna jest bez przebudowywania.

# Załaduj workspace
source install/setup.bash
```

### Plik package.xml – po co i jak go czytać

```xml
<?xml version="1.0"?>
<package format="3">
  <!-- Nazwa pakietu – musi być unikalna w ekosystemie ROS2 -->
  <name>my_robot_pkg</name>
  <version>0.0.1</version>
  <description>Przykładowy pakiet robota</description>
  <maintainer email="student@prz.edu.pl">Student PRz</maintainer>
  <license>Apache-2.0</license>

  <!-- buildtool_depend: narzędzia potrzebne do BUDOWANIA (nie do działania) -->
  <buildtool_depend>ament_python</buildtool_depend>

  <!-- depend: zależności potrzebne zarówno do budowania jak i działania -->
  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <!-- test_depend: tylko do testów jednostkowych -->
  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Podstawowy Przykład: Publisher i Subscriber

### Publisher Node (Python) – szczegółowe omówienie

Node Publisher to węzeł wysyłający wiadomości na temat (topic). Poniżej znajdziesz kod z dokładnymi wyjaśnieniami każdej linii.

```python
import rclpy                          # Główna biblioteka ROS2 dla Pythona (rclpy = ROS Client Library Python)
from rclpy.node import Node           # Klasa bazowa dla wszystkich węzłów ROS2
from std_msgs.msg import String       # Typ wiadomości: standardowa wiadomość tekstowa
                                      # std_msgs to pakiet ze standardowymi typami danych

class MinimalPublisher(Node):
    # Dlaczego dziedziczymy po Node?
    # Node to klasa bazowa dostarczająca całą infrastrukturę ROS2:
    # logowanie, tworzenie publisherów/subskrybentów, timery, parametry itp.
    # Dziedziczenie pozwala naszej klasie "być" węzłem ROS2.

    def __init__(self):
        super().__init__('minimal_publisher')
        # super().__init__('minimal_publisher') – wywołujemy konstruktor klasy Node.
        # Argument 'minimal_publisher' to NAZWA WĘZŁA widoczna w grafie ROS2.
        # Nazwa musi być unikalna w systemie – jeśli uruchomimy dwa węzły
        # o tej samej nazwie, drugi zastąpi pierwszy (lub oba będą działać
        # zależnie od konfiguracji DDS).

        self.publisher_ = self.create_publisher(String, 'topic', 10)
        # create_publisher(typ, nazwa_topicu, qos_depth)
        # - String: typ wiadomości – musi być zgodny z tym czego oczekuje Subscriber
        # - 'topic': nazwa topicu – to "adres" na który wysyłamy wiadomości
        #   Dobra praktyka: używaj opisowych nazw, np. '/robot/status', '/camera/image'
        # - 10: głębokość kolejki QoS (Queue Depth)
        #   Jeśli system jest przeciążony i nie nadąża przetwarzać wiadomości,
        #   w buforze zostanie max 10 ostatnich. Starsze zostaną odrzucone.
        #   Mała wartość (1-5) = aktualność danych; duża (100+) = kompletność danych.

        self.timer = self.create_timer(0.5, self.timer_callback)
        # create_timer(okres_w_sekundach, funkcja_callback)
        # - 0.5 sekundy = 2 Hz (dwa razy na sekundę)
        # - self.timer_callback: funkcja wywoływana co 0.5 s
        # Dlaczego timer zamiast while True + sleep()?
        # Timer ROS2 jest zarządzany przez executor – może być precyzyjniej
        # synchronizowany z innymi callbackami, nie blokuje wątku,
        # a ROS2 może go zatrzymać podczas shutdown.

        self.i = 0
        # Licznik wiadomości – żeby każda wiadomość była unikalna.
        # Przydatne przy debugowaniu: widzisz że wiadomości są kolejno numerowane
        # i możesz wykryć "dziury" (utracone wiadomości).

    def timer_callback(self):
        # Ta metoda jest wywoływana automatycznie przez ROS2 co 0.5 sekundy.

        msg = String()
        # Tworzymy obiekt wiadomości. Zawsze tworzymy nowy obiekt – nie modyfikujemy
        # starego. Gdybyśmy modyfikowali stary obiekt po publish(), mogłoby dojść
        # do wyścigu (race condition) jeśli ROS2 jeszcze go przetwarza.

        msg.data = f'Hello World: {self.i}'
        # Wypełniamy pole 'data' wiadomości String.
        # Typ String z std_msgs ma tylko jedno pole: 'data' typu string.
        # Inne typy wiadomości mają więcej pól (np. geometry_msgs/Pose
        # ma position.x, position.y, position.z, orientation.x itd.)

        self.publisher_.publish(msg)
        # Wysyłamy wiadomość. Metoda publish() jest nieblokująca –
        # przekazuje wiadomość do DDS i wraca natychmiast.
        # DDS zajmuje się dostarczeniem wiadomości do wszystkich Subscriberów.

        self.get_logger().info(f'Publishing: "{msg.data}"')
        # get_logger() zwraca logger ROS2 dla tego węzła.
        # .info() loguje wiadomość na poziomie INFO (widoczna w terminalu).
        # Poziomy logowania: DEBUG < INFO < WARN < ERROR < FATAL
        # Dlaczego nie print()? Logger ROS2 obsługuje filtrowanie poziomów,
        # znaczniki czasu, można go przekierować do pliku lub /rosout topic.

        self.i += 1

def main(args=None):
    # args=None pozwala na przekazanie argumentów z linii poleceń
    # (np. __name:=nowa_nazwa do przemianowania węzła przy uruchamianiu)

    rclpy.init(args=args)
    # Inicjalizacja biblioteki rclpy – musi być wywołana PRZED stworzeniem
    # jakiegokolwiek węzła. Inicjalizuje komunikację DDS, rejestruje
    # kontekst wykonania.

    node = MinimalPublisher()
    # Tworzymy instancję węzła.

    rclpy.spin(node)
    # spin() uruchamia pętlę zdarzeń (event loop).
    # Czeka na callbacki (timery, wiadomości, serwisy) i je wywołuje.
    # Blokuje wykonanie do momentu gdy węzeł zostanie zatrzymany
    # (np. przez Ctrl+C lub rclpy.shutdown()).
    # Dlaczego spin() zamiast własnej pętli? spin() jest zintegrowany
    # z DDS – pozwala na prawidłową obsługę priorytetów i synchronizację.

    node.destroy_node()
    # Czyścimy zasoby węzła: anuluje timery, usuwa publishery/subscribery,
    # wyrejestruje węzeł z sieci DDS.

    rclpy.shutdown()
    # Zamykamy bibliotekę rclpy i DDS.
    # Kolejność destroy_node() → rclpy.shutdown() jest ważna –
    # odwrotna kolejność może powodować błędy przy zamykaniu.
```

### Subscriber Node (Python) – szczegółowe omówienie

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')

        self.subscription = self.create_subscription(
            String,          # Typ wiadomości – MUSI być taki sam jak w Publisherze!
                             # Niezgodność typów → brak komunikacji (ROS2 nie połączy)
            'topic',         # Nazwa topicu – ta sama co w Publisherze
            self.listener_callback,  # Funkcja wywoływana gdy przyjdzie wiadomość
            10)              # Głębokość kolejki QoS – jak w Publisherze
        # Uwaga: wynik create_subscription() zapisujemy do self.subscription.
        # Gdybyśmy tego nie zrobili, Python usunąłby obiekt z pamięci (garbage collector)
        # i subskrypcja przestałaby działać! To częsty błąd początkujących.

    def listener_callback(self, msg):
        # Ta metoda jest wywoływana automatycznie przez ROS2 za każdym razem
        # gdy na topicu 'topic' pojawi się nowa wiadomość.
        # Argument 'msg' to obiekt typu String z polem 'data'.

        self.get_logger().info(f'I heard: "{msg.data}"')
        # Logujemy odebraną wiadomość.
        # W prawdziwej aplikacji tutaj byłaby logika przetwarzania:
        # - parsowanie danych z sensora
        # - aktualizacja stanu robota
        # - obliczenia i publikacja wyników

def main(args=None):
    rclpy.init(args=args)
    node = MinimalSubscriber()
    rclpy.spin(node)           # Czekamy na wiadomości w pętli zdarzeń
    node.destroy_node()
    rclpy.shutdown()
```

### Jak uruchomić Publisher i Subscriber?

```bash
# Terminal 1: uruchamiamy Publisher
source ~/ros2_ws/install/setup.bash
ros2 run my_robot_pkg minimal_publisher

# Terminal 2: uruchamiamy Subscriber
source ~/ros2_ws/install/setup.bash
ros2 run my_robot_pkg minimal_subscriber

# Terminal 3: weryfikacja – sprawdzamy czy topic istnieje i co jest na nim wysyłane
ros2 topic list                    # Pokaże /topic na liście
ros2 topic echo /topic             # Wyświetla wiadomości w czasie rzeczywistym
ros2 topic hz /topic               # Mierzy częstotliwość wiadomości (powinno być ~2 Hz)
ros2 topic info /topic             # Typ wiadomości, liczba pub/sub
```

## Services – Komunikacja Synchroniczna

Serwisy używamy gdy potrzebujemy odpowiedzi – np. "czy robot jest gotowy?", "podaj aktualną pozycję", "wykonaj kalibrację". W odróżnieniu od topics (asynchroniczne, "fire and forget"), serwis czeka na odpowiedź.

### Kiedy używać Service zamiast Topic?

| Cecha | Topic | Service |
|-------|-------|---------|
| Model komunikacji | Publish-Subscribe | Request-Response |
| Czekanie na odpowiedź | NIE | TAK |
| Użycie | Strumień danych (np. kamera, IMU) | Jednorazowe zapytanie |
| Wiele odbiorców | TAK (broadcast) | NIE (jeden serwer) |
| Przykład | dane z czujnika, pozycja | komenda kalibracji, zapytanie o stan |

### Definicja własnego serwisu (.srv)

Zanim napiszemy kod, definiujemy interfejs serwisu w pliku `.srv`:

```
# Plik: srv/AddTwoInts.srv
# Sekcja Request (co klient wysyła)
int64 a
int64 b
---
# Trzy myślniki oddzielają Request od Response
# Sekcja Response (co serwer odsyła)
int64 sum
```

### Serwer Serwisu (Service Server)

```python
from example_interfaces.srv import AddTwoInts  # importujemy interfejs serwisu
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')

        self.srv = self.create_service(
            AddTwoInts,          # Typ serwisu (musi pasować do klienta)
            'add_two_ints',      # Nazwa serwisu – "adres" pod którym nasłuchujemy
            self.add_two_ints_callback)  # Funkcja obsługująca żądania
        # Serwer jest gotowy do przyjmowania żądań od razu po create_service().

    def add_two_ints_callback(self, request, response):
        # request: obiekt z polami zdefiniowanymi w sekcji Request pliku .srv
        # response: obiekt który MUSIMY wypełnić i zwrócić

        response.sum = request.a + request.b
        # Wypełniamy pole 'sum' w odpowiedzi.
        # Robimy to SYNCHRONICZNIE – callback musi zakończyć się jak najszybciej,
        # bo klient czeka zablokowany! Długie obliczenia należy przenieść
        # do osobnego wątku lub użyć Actions.

        self.get_logger().info(
            f'Incoming request: a={request.a}, b={request.b} → sum={response.sum}')

        return response
        # MUSIMY zwrócić response – inaczej klient nie dostanie odpowiedzi!

def main():
    rclpy.init()
    node = MinimalService()
    rclpy.spin(node)
    rclpy.shutdown()
```

### Klient Serwisu (Service Client)

```python
import sys
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')

        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        # Tworzymy klienta dla serwisu 'add_two_ints'.
        # W tym momencie klient SZUKA serwera w sieci DDS.
        # Jeśli serwer nie istnieje, klient będzie czekać.

        while not self.cli.wait_for_service(timeout_sec=1.0):
            # Czekamy aż serwis będzie dostępny.
            # timeout_sec=1.0 → sprawdzamy co sekundę i wypisujemy komunikat.
            # Dlaczego pętla? Serwer może startować wolniej niż klient –
            # szczególnie w systemach z wieloma węzłami startującymi jednocześnie.
            self.get_logger().info('Serwis niedostępny, czekam...')

        self.req = AddTwoInts.Request()
        # Tworzymy obiekt żądania. Pola wypełnimy przed wysłaniem.

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b

        self.future = self.cli.call_async(self.req)
        # call_async() wysyła żądanie ASYNCHRONICZNIE i zwraca obiekt Future.
        # Dlaczego asynchronicznie? Synchroniczne call() blokowałoby event loop,
        # uniemożliwiając przetwarzanie innych callbacków (np. subskrypcji).
        # Future to "obietnica" wyniku – sprawdzamy go później.

        rclpy.spin_until_future_complete(self, self.future)
        # Obracamy event loop AŻ DO momentu gdy Future zostanie wypełniony
        # (tj. serwer wyśle odpowiedź). To blokuje, ale pozwala na przetwarzanie
        # innych callbacków w międzyczasie.

        return self.future.result()
        # Zwracamy wynik (obiekt Response z polem 'sum').

def main(args=None):
    rclpy.init(args=args)
    client = MinimalClient()

    result = client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    # Przekazujemy argumenty z linii poleceń jako liczby.
    # Uruchomienie: ros2 run my_pkg client 3 7  → wynik: 10

    client.get_logger().info(f'Wynik: {result.sum}')
    client.destroy_node()
    rclpy.shutdown()
```

## Actions – Długotrwałe Zadania

Actions to najbardzej złożony mechanizm komunikacji w ROS2. Używamy ich gdy:
- Zadanie trwa **długo** (sekundy, minuty)
- Potrzebujemy **śledzenia postępu** (feedback co X sekund)
- Chcemy móc **anulować** zadanie w trakcie

Przykłady: nawigacja do punktu, podnoszenie obiektu, kalibracja czujnika.

### Definicja interfejsu Action (.action)

```
# Plik: action/Navigate.action
# --- GOAL: cel zadania ---
geometry_msgs/PoseStamped target_pose   # dokąd robot ma pojechać
---
# --- RESULT: wynik końcowy (po zakończeniu) ---
geometry_msgs/PoseStamped final_pose    # gdzie faktycznie dotarł
float32 total_elapsed_time             # ile czasu zajęła nawigacja
---
# --- FEEDBACK: postęp (wysyłany wielokrotnie w trakcie) ---
geometry_msgs/PoseStamped current_pose  # aktualna pozycja
float32 distance_remaining             # ile zostało do celu
```

### Action Server

```python
import time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci  # użyjemy gotowego przykładu

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')

        self._action_server = ActionServer(
            self,
            Fibonacci,              # typ action (z pliku .action)
            'fibonacci',            # nazwa action
            self.execute_callback)  # główna funkcja wykonująca zadanie
        # W odróżnieniu od Service, Action Server obsługuje:
        # - przyjęcie celu (goal)
        # - wysyłanie feedbacku w trakcie
        # - możliwość anulowania przez klienta

    def execute_callback(self, goal_handle):
        # goal_handle: obiekt zarządzający bieżącym zadaniem
        # goal_handle.request → parametry zadania (z sekcji GOAL)

        self.get_logger().info(f'Obliczam Fibonacci({goal_handle.request.order})...')

        feedback_msg = Fibonacci.Feedback()
        # Tworzymy obiekt feedbacku – będziemy go wysyłać co krok.

        sequence = [0, 1]  # pierwsze dwa wyrazy ciągu Fibonacciego

        for i in range(1, goal_handle.request.order):
            # Obliczamy kolejne wyrazy ciągu Fibonacciego
            sequence.append(sequence[i] + sequence[i-1])

            feedback_msg.partial_sequence = sequence
            goal_handle.publish_feedback(feedback_msg)
            # Wysyłamy feedback do klienta – klient widzi postęp obliczeń
            # na bieżąco, zanim zadanie się zakończy.

            # Sprawdzamy czy klient nie anulował zadania
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Zadanie anulowane przez klienta')
                return Fibonacci.Result()
                # Gdy klient anuluje → przerywamy i zwracamy pusty wynik.

            time.sleep(0.1)  # Symulujemy długotrwałe obliczenia

        goal_handle.succeed()
        # Oznaczamy zadanie jako zakończone sukcesem.
        # Alternatywy: goal_handle.abort() gdy coś pójdzie nie tak.

        result = Fibonacci.Result()
        result.sequence = sequence
        return result
        # Zwracamy finalny wynik (sekcja RESULT).

def main():
    rclpy.init()
    server = FibonacciActionServer()
    rclpy.spin(server)
    rclpy.shutdown()
```

### Action Client

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        # Czekamy na dostępność serwera (analogicznie do Service)
        self._action_client.wait_for_server()

        goal_msg = Fibonacci.Goal()
        goal_msg.order = order  # chcemy n-ty wyraz Fibonacciego

        # Wysyłamy cel ASYNCHRONICZNIE – callback zostanie wywołany gdy
        # serwer zaakceptuje (lub odrzuci) cel.
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)  # callback dla feedbacku

        self._send_goal_future.add_done_callback(self.goal_response_callback)
        # Gdy serwer odpowie na żądanie celu → wywołaj goal_response_callback.

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Cel odrzucony przez serwer!')
            return
            # Serwer może odrzucić cel np. gdy jest już zajęty innym zadaniem.

        self.get_logger().info('Cel zaakceptowany, czekam na wynik...')

        # Pobieramy wynik ASYNCHRONICZNIE
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        # Wywoływany wielokrotnie w trakcie wykonywania zadania.
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Dotychczasowy ciąg: {feedback.partial_sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Finalny wynik: {result.sequence}')
        rclpy.shutdown()

def main():
    rclpy.init()
    client = FibonacciActionClient()
    client.send_goal(10)  # Oblicz 10 wyrazów ciągu Fibonacciego
    rclpy.spin(client)    # Czekamy na zakończenie (feedbacki + wynik końcowy)
```

## Quality of Service (QoS)

QoS pozwala dostosować zachowanie komunikacji do wymagań aplikacji. Jest to jeden z największych atutów ROS2 nad ROS1.

### Główne parametry QoS

```python
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,   # czy każda wiadomość MUSI dotrzeć?
    HistoryPolicy,       # ile wiadomości przechowywać w buforze?
    DurabilityPolicy,    # czy nowy Subscriber dostaje stare wiadomości?
    LivelinessPolicy,    # jak wykrywać "martwe" węzły?
)

# --- Profil dla danych czujnikowych (kamera, LiDAR) ---
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    # BEST_EFFORT: wysyłamy co mamy, bez retransmisji.
    # Dlaczego dla sensorów? Dane z kamery są generowane 30x/s.
    # Jeśli jedna klatka się zgubi – nic strasznego, zaraz przyjdzie następna.
    # RELIABLE (z retransmisjami) spowodowałby "zakorkowanie" sieci
    # i zwiększone opóźnienia – niedopuszczalne dla real-time.

    history=HistoryPolicy.KEEP_LAST,
    # KEEP_LAST: trzymamy tylko N ostatnich wiadomości (N = depth).
    # KEEP_ALL: trzymamy WSZYSTKIE – niebezpieczne dla szybkich sensorów
    # bo może prowadzić do nieskończonego wzrostu pamięci.

    depth=1,
    # depth=1 dla sensorów: zawsze pracujemy na NAJNOWSZYCH danych.
    # Nie ma sensu przetwarzać starych klatek z kamery.
)

# --- Profil dla krytycznych komend sterowania ---
control_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    # RELIABLE: gwarantuje dostarczenie – używa retransmisji.
    # Dla komend sterowania silnikiem: każda komenda MUSI dotrzeć.
    # Utracona komenda = robot nie wykona ruchu = błąd bezpieczeństwa.

    history=HistoryPolicy.KEEP_LAST,
    depth=10,

    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    # TRANSIENT_LOCAL: gdy nowy Subscriber się podłączy,
    # Publisher wysyła mu ostatnią wiadomość od razu (zamiast czekać na nową).
    # Przydatne dla "statusów" – np. stan naładowania baterii:
    # nowy węzeł od razu zna aktualny stan, nie musi czekać na aktualizację.
)

# Użycie profilu:
self.publisher_ = self.create_publisher(String, 'topic', sensor_qos)
self.subscription = self.create_subscription(String, 'topic', self.cb, control_qos)
```

### Kompatybilność QoS

> **Ważne!** Publisher i Subscriber muszą mieć **kompatybilne** profile QoS, inaczej nie będą się komunikować. Zasada: Subscriber może mieć "słabsze" wymagania niż Publisher, ale nie silniejsze.

```
Publisher: RELIABLE  ←→  Subscriber: RELIABLE   ✓ OK
Publisher: RELIABLE  ←→  Subscriber: BEST_EFFORT ✓ OK (degradacja)
Publisher: BEST_EFFORT ←→ Subscriber: RELIABLE   ✗ BŁĄD – brak komunikacji!
```

## Parametry Węzła

Parametry pozwalają konfigurować węzeł bez modyfikacji kodu i przebudowywania. Są przechowywane w węźle i mogą być zmieniane w czasie działania.

```python
import rclpy
from rclpy.node import Node

class ParametrizedNode(Node):
    def __init__(self):
        super().__init__('parametrized_node')

        # Deklaracja parametrów z wartościami domyślnymi.
        # WAŻNE: parametr MUSI być zadeklarowany przed użyciem.
        # Deklaracja to "umowa" – mówi ROS2 jaki typ i wartość domyślna.
        self.declare_parameter('robot_name', 'unitree_g1')
        # Typ automatycznie wykrywany z wartości domyślnej: str → ParameterType.STRING

        self.declare_parameter('publish_frequency', 10.0)
        # float → ParameterType.DOUBLE

        self.declare_parameter('max_velocity', 1.5)
        # Parametr konfiguracyjny – różne instancje robota mogą mieć różne
        # maksymalne prędkości bez zmiany kodu.

        # Odczytanie wartości parametrów
        robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        freq = self.get_parameter('publish_frequency').get_parameter_value().double_value
        max_vel = self.get_parameter('max_velocity').get_parameter_value().double_value

        self.get_logger().info(
            f'Robot: {robot_name}, freq: {freq} Hz, max_vel: {max_vel} m/s')

        # Timer z dynamiczną częstotliwością z parametru
        self.timer = self.create_timer(1.0 / freq, self.timer_callback)

        # Callback wywoływany gdy parametr zostanie zmieniony z zewnątrz
        self.add_on_set_parameters_callback(self.parameters_callback)

    def parameters_callback(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for param in params:
            if param.name == 'publish_frequency' and param.value > 0:
                # Dynamiczna aktualizacja timera po zmianie częstotliwości
                self.timer.cancel()
                self.timer = self.create_timer(
                    1.0 / param.value, self.timer_callback)
                self.get_logger().info(f'Zmieniono częstotliwość na {param.value} Hz')
        return SetParametersResult(successful=True)
        # Zwracamy True = akceptujemy zmianę.
        # Gdybyśmy zwrócili False, parametr nie zostanie zmieniony.

    def timer_callback(self):
        self.get_logger().info('Tick!')

def main():
    rclpy.init()
    node = ParametrizedNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

```bash
# Uruchomienie z nadpisaniem parametrów z linii poleceń
ros2 run my_pkg parametrized_node --ros-args \
  -p robot_name:=g1_lab_01 \
  -p publish_frequency:=50.0 \
  -p max_velocity:=0.5

# Zmiana parametru w trakcie działania węzła
ros2 param set /parametrized_node max_velocity 1.0

# Odczyt aktualnej wartości parametru
ros2 param get /parametrized_node robot_name

# Lista wszystkich parametrów węzła
ros2 param list /parametrized_node
```

## TF2 – System Transformacji Układów Współrzędnych

TF2 (Transform Library 2) to jeden z najważniejszych podsystemów ROS2. Pozwala śledzić relacje geometryczne między układami współrzędnych w czasie.

**Dlaczego TF2?** Robot ma wiele układów współrzędnych:
- `world` – globalny układ świata
- `base_link` – środek robota
- `camera_frame` – układ kamery
- `lidar_frame` – układ LiDAR-a
- `left_hand` – układ lewej dłoni

TF2 automatycznie śledzi jak te układy poruszają się względem siebie i umożliwia przeliczanie pozycji między nimi.

```python
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
import math

class TF2BroadcasterExample(Node):
    """Przykład: nadajnik transformacji (broadcaster)."""
    def __init__(self):
        super().__init__('tf2_broadcaster')

        # StaticTransformBroadcaster – dla transformacji NIEZMIENNYCH w czasie
        # (np. montaż kamery na ramieniu – zawsze w tym samym miejscu)
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # TransformBroadcaster – dla transformacji ZMIENNYCH w czasie
        # (np. pozycja podstawy robota w świecie, zmienia się gdy robot jedzie)
        self.dynamic_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Wysyłamy statyczną transformację: kamera jest 0.1m przed i 0.3m nad base_link
        self.send_static_transform()

        # Timer do wysyłania dynamicznych transformacji
        self.timer = self.create_timer(0.1, self.send_dynamic_transform)
        self.angle = 0.0

    def send_static_transform(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        # Znacznik czasu – ZAWSZE używamy zegara ROS2 (nie Python time.time()!)
        # Dlaczego? Zegar ROS2 może być przyspieszony w symulacji, a TF2
        # interpoluje transformacje po czasie → poprawny czas jest kluczowy.

        t.header.frame_id = 'base_link'     # układ RODZICA
        t.child_frame_id = 'camera_frame'   # układ DZIECKA

        # Translacja: kamera 10 cm przed i 30 cm nad centrum robota
        t.transform.translation.x = 0.1
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.3

        # Rotacja jako kwaternion (nie kąty Eulera!)
        # Dlaczego kwaterniony? Nie mają problemu gimbal lock,
        # łatwo interpolować sferyczną interpolacją (SLERP).
        # Brak obrotu = kwaternion jednostkowy [0, 0, 0, 1]
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.static_broadcaster.sendTransform(t)

    def send_dynamic_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'

        # Robot porusza się po okręgu (symulacja)
        t.transform.translation.x = math.cos(self.angle)
        t.transform.translation.y = math.sin(self.angle)
        t.transform.translation.z = 0.0

        # Obrót wokół osi Z = kierunek ruchu robota
        t.transform.rotation.z = math.sin(self.angle / 2)
        t.transform.rotation.w = math.cos(self.angle / 2)

        self.dynamic_broadcaster.sendTransform(t)
        self.angle += 0.01

class TF2ListenerExample(Node):
    """Przykład: odbiorca transformacji (listener)."""
    def __init__(self):
        super().__init__('tf2_listener')

        self.tf_buffer = tf2_ros.Buffer()
        # Buffer przechowuje historię transformacji (domyślnie ostatnie 10 sekund).
        # To pozwala na interpolację – "gdzie był obiekt 50ms temu?"

        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # TransformListener subskrybuje /tf i /tf_static i wypełnia buffer.

        self.timer = self.create_timer(1.0, self.on_timer)

    def on_timer(self):
        try:
            # Pobieramy transformację: "gdzie jest camera_frame względem world?"
            # rclpy.time.Time() = "najnowsza dostępna transformacja"
            transform = self.tf_buffer.lookup_transform(
                'world',          # układ docelowy (do którego przeliczamy)
                'camera_frame',   # układ źródłowy (który przeliczamy)
                rclpy.time.Time())

            pos = transform.transform.translation
            self.get_logger().info(
                f'Kamera w świecie: x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f}')

        except tf2_ros.TransformException as ex:
            # Transformacja niedostępna – np. broadcaster jeszcze nie wysłał
            # lub upłynął timeout. ZAWSZE obsługujemy ten wyjątek!
            self.get_logger().warn(f'Transformacja niedostępna: {ex}')
```

## Launch Files – Uruchamianie Systemu

Launch files pozwalają uruchomić cały złożony system (wiele węzłów, konfiguracje, warunki) jednym poleceniem.

```python
# Plik: launch/robot_system.launch.py
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,    # deklaracja argumentów z linii poleceń
    IncludeLaunchDescription, # dołączanie innych plików launch
    ExecuteProcess,           # uruchamianie procesów systemowych (nie węzłów ROS2)
    GroupAction,              # grupowanie akcji
)
from launch.conditions import IfCondition, UnlessCondition
# Warunki pozwalają uruchamiać węzły warunkowo – np. tylko w trybie debug.

from launch.substitutions import (
    LaunchConfiguration,  # odczytanie argumentu
    PythonExpression,     # ewaluacja wyrażenia Python w launch file
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # --- Deklaracja argumentów ---
    # Argumenty pozwalają użytkownikowi dostosować zachowanie launch file
    # bez jego modyfikacji.

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',     # nazwa argumentu
        default_value='false',  # wartość domyślna
        description='Używaj czasu symulacji zamiast czasu rzeczywistego'
        # Czas symulacji pozwala odtwarzać nagrania rosbag w innej prędkości.
    )

    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='unitree_g1',
        description='Nazwa robota'
    )

    enable_debug_arg = DeclareLaunchArgument(
        'debug', default_value='false',
        description='Włącz węzły debugowania'
    )

    use_sim = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')

    # --- Węzły ---

    publisher_node = Node(
        package='my_robot_pkg',       # nazwa pakietu ROS2
        executable='minimal_publisher', # nazwa entry point z setup.py
        name='publisher_node',        # nadpisanie nazwy węzła (opcjonalne)
        output='screen',              # gdzie wyświetlać logi: 'screen' lub 'log'
        # 'screen' → terminal, 'log' → plik w ~/.ros/log/
        parameters=[{
            'use_sim_time': use_sim,  # przekazujemy argument launch do parametru węzła
            'publish_frequency': 10.0,
        }],
        remappings=[
            # Remapping pozwala zmienić nazwy topics/serwisów bez modyfikacji kodu.
            # Publisher wysyła na 'topic', ale my chcemy '/robot/status'
            ('topic', '/robot/status'),
        ]
    )

    subscriber_node = Node(
        package='my_robot_pkg',
        executable='minimal_subscriber',
        name='subscriber_node',
        output='screen',
        remappings=[('topic', '/robot/status')]
        # Subscriber odbiera z 'topic' → remappujemy na '/robot/status'
        # Teraz publisher i subscriber automatycznie się znajdą.
    )

    # Węzeł debugowania – uruchamiany TYLKO gdy debug:=true
    debug_node = Node(
        package='my_robot_pkg',
        executable='debug_monitor',
        name='debug_monitor',
        condition=IfCondition(LaunchConfiguration('debug'))
        # IfCondition → węzeł uruchamia się gdy debug=true
        # UnlessCondition → węzeł uruchamia się gdy debug=false
    )

    return LaunchDescription([
        use_sim_time_arg,
        robot_name_arg,
        enable_debug_arg,
        publisher_node,
        subscriber_node,
        debug_node,
    ])
```

```bash
# Uruchomienie launch file
ros2 launch my_robot_pkg robot_system.launch.py

# Z argumentami
ros2 launch my_robot_pkg robot_system.launch.py use_sim_time:=true debug:=true

# Podgląd wszystkich dostępnych argumentów launch file
ros2 launch my_robot_pkg robot_system.launch.py --show-args
```

## Własne Typy Wiadomości

Standardowe typy (std_msgs, geometry_msgs) często nie wystarczają. Możemy definiować własne.

```
# Plik: msg/RobotStatus.msg
# Własna wiadomość agregująca stan robota

std_msgs/Header header    # znacznik czasu i frame_id (konwencja ROS2)
string robot_name         # identyfikator robota
float32 battery_level     # 0.0 - 100.0 [%]
float32[] joint_positions # tablica pozycji stawów [rad]
bool is_operational       # czy robot jest sprawny
uint8 error_code          # kod błędu (0 = brak błędu)

# Stałe (konwencja: WIELKIE_LITERY)
uint8 ERROR_NONE=0
uint8 ERROR_LOW_BATTERY=1
uint8 ERROR_JOINT_LIMIT=2
uint8 ERROR_COMM_TIMEOUT=3
```

```python
# Użycie własnej wiadomości w węźle
from my_robot_pkg.msg import RobotStatus  # importujemy z naszego pakietu
from std_msgs.msg import Header
import rclpy
from rclpy.node import Node

class RobotStatusPublisher(Node):
    def __init__(self):
        super().__init__('robot_status_pub')
        self.pub = self.create_publisher(RobotStatus, '/robot/status', 10)
        self.timer = self.create_timer(1.0, self.publish_status)

    def publish_status(self):
        msg = RobotStatus()

        # Wypełniamy header – konwencja ROS2: zawsze wypełniaj header!
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        # frame_id = w jakim układzie współrzędnych są dane przestrzenne.
        # Dla statusu (nie pozycji) możemy podać 'base_link' lub zostawić puste.

        msg.robot_name = 'unitree_g1'
        msg.battery_level = 85.5
        msg.joint_positions = [0.0, 0.1, -0.2, 0.3, 0.0, 0.0]  # 6 stawów
        msg.is_operational = True
        msg.error_code = RobotStatus.ERROR_NONE  # używamy stałej, nie liczby

        self.pub.publish(msg)
```

```bash
# Musimy poinformować package.xml i CMakeLists.txt/setup.py o nowej wiadomości.
# Następnie przebudować pakiet:
cd ~/ros2_ws
colcon build --packages-select my_robot_pkg
source install/setup.bash

# Podgląd definicji wiadomości
ros2 interface show my_robot_pkg/msg/RobotStatus
```

## Użycie w Laboratorium

W naszym laboratorium ROS2 Humble jest podstawowym frameworkiem do:
- Komunikacji między modułami robota Unitree G1
- Integracji sensorów (LiDAR, kamery, mikrofony)
- Sterowania manipulatorami Dex3-1
- Integracji z systemami AI (PyTorch, TensorFlow)

### Przykład: Integracja kamery z modelem AI

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image       # standardowa wiadomość ROS2 dla obrazów
from std_msgs.msg import String
from cv_bridge import CvBridge          # konwersja między ROS2 Image a OpenCV Mat
import cv2
import numpy as np

class VisionPipelineNode(Node):
    """
    Węzeł przetwarzania obrazu z kamery.
    Subskrybuje obraz → przetwarza → publikuje wyniki detekcji.
    """
    def __init__(self):
        super().__init__('vision_pipeline')

        # CvBridge: konwertuje sensor_msgs/Image ↔ numpy array (OpenCV format)
        # Dlaczego bridge? ROS2 przechowuje obraz jako bajty + metadane (szerokość,
        # wysokość, encoding). OpenCV używa numpy array. Bridge obsługuje konwersję.
        self.bridge = CvBridge()

        # Subskrybujemy strumień obrazu z kamery
        # Używamy QoS sensor_data (BEST_EFFORT) – odpowiedni dla kamer
        from rclpy.qos import qos_profile_sensor_data
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',   # temat publikowany przez sterownik kamery RealSense
            self.image_callback,
            qos_profile_sensor_data)

        # Publikujemy wyniki detekcji
        self.result_pub = self.create_publisher(String, '/vision/detections', 10)

        self.get_logger().info('Węzeł wizji uruchomiony, oczekuję na obrazy...')

    def image_callback(self, msg):
        # Konwersja ROS2 Image → numpy array BGR (format OpenCV)
        # 'bgr8' = Blue-Green-Red, 8 bitów na kanał (typowy format dla kamer USB)
        # Inne enkodowania: 'rgb8', 'mono8' (czarno-biały), '16UC1' (depth map)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Błąd konwersji obrazu: {e}')
            return

        # Tutaj umieszczamy logikę przetwarzania/detekcji
        # Przykład: prosta detekcja krawędzi Canny
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_count = np.count_nonzero(edges)

        # Publikujemy wynik
        result_msg = String()
        result_msg.data = f'Wykryto {edge_count} pikseli krawędzi'
        self.result_pub.publish(result_msg)

def main():
    rclpy.init()
    node = VisionPipelineNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

## Narzędzia CLI

Narzędzia CLI (Command Line Interface) ROS2 są niezbędne do debugowania i monitorowania systemu.

```bash
# ==========================================
# WĘZŁY (nodes)
# ==========================================

# Lista aktywnych węzłów
ros2 node list
# Wyświetla: /minimal_publisher, /minimal_subscriber, /robot_status_pub itp.

# Szczegółowe informacje o węźle
ros2 node info /minimal_publisher
# Wyświetla: publishers, subscribers, services, actions, parameters węzła.
# Przydatne gdy chcemy sprawdzić czy węzeł "widzi" właściwe topics.

# ==========================================
# TOPICS (tematy)
# ==========================================

# Lista aktywnych topics
ros2 topic list
# Pokazuje wszystkie aktywne topics. -t dodaje typ wiadomości.
ros2 topic list -t

# Podgląd wiadomości w czasie rzeczywistym
ros2 topic echo /topic
ros2 topic echo /robot/status        # własny typ wiadomości
ros2 topic echo /camera/color/image_raw --no-arr  # --no-arr ukrywa duże tablice

# Częstotliwość publikacji (Hz)
ros2 topic hz /topic
# Wyświetla min/max/mean/stddev częstotliwości. Przydatne do weryfikacji
# czy węzeł pracuje z oczekiwaną częstotliwością.

# Opóźnienie (latency) wiadomości
ros2 topic delay /topic
# Mierzy różnicę między znacznikiem czasu w header a aktualnym czasem.
# UWAGA: działa tylko dla wiadomości z polem 'header' (std_msgs/Header).

# Informacje o topicu
ros2 topic info /topic
# Pokazuje typ, liczbę publisherów i subskrybentów.
# Jeśli pub=1, sub=0 → nikt nie odbiera danych (może być błąd w remappingu).

# Ręczne publikowanie wiadomości z terminala
ros2 topic pub /topic std_msgs/msg/String "data: 'Hello from terminal'" --once
ros2 topic pub --rate 1 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}"
# --rate 1 → publikuj co sekundę (do Ctrl+C)
# --once → opublikuj jedną wiadomość i zakończ

# ==========================================
# SERVICES (serwisy)
# ==========================================

ros2 service list                           # lista aktywnych serwisów
ros2 service list -t                        # z typami
ros2 service type /add_two_ints             # typ konkretnego serwisu
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 3, b: 7}"
# Wywołuje serwis i wyświetla odpowiedź. Przydatne do testowania bez pisania klienta.

# ==========================================
# PARAMETRY (parameters)
# ==========================================

ros2 param list                             # parametry wszystkich węzłów
ros2 param list /minimal_publisher          # parametry konkretnego węzła
ros2 param get /minimal_publisher use_sim_time
ros2 param set /minimal_publisher publish_frequency 20.0
ros2 param dump /minimal_publisher          # dump do YAML (do pliku konfiguracyjnego)
ros2 param load /minimal_publisher params.yaml  # wczytaj z pliku

# ==========================================
# NAGRYWANIE I ODTWARZANIE (rosbag2)
# ==========================================

# Nagrywanie WSZYSTKICH topics
ros2 bag record -a
# -a (all) nagrywa wszystko. Można podać wybrane topics:
ros2 bag record /camera/color/image_raw /robot/status /tf

# Odtwarzanie nagrania
ros2 bag play ./rosbag2_2024_01_15-10_30_00/
# --rate 0.5 → odtwarzaj 2x wolniej (przydatne do analizy)
ros2 bag play ./rosbag2_2024_01_15/ --rate 0.5

# Informacje o nagraniu
ros2 bag info ./rosbag2_2024_01_15/
# Wyświetla: czas trwania, rozmiar, lista topics, liczba wiadomości.

# ==========================================
# INTERFEJSY (interfaces)
# ==========================================

ros2 interface list                         # wszystkie dostępne typy
ros2 interface show std_msgs/msg/String     # definicja wiadomości
ros2 interface show example_interfaces/srv/AddTwoInts  # definicja serwisu
ros2 interface show nav2_msgs/action/NavigateToPose    # definicja action

# ==========================================
# DIAGNOSTYKA SYSTEMU
# ==========================================

# Graf komunikacji (GUI)
rqt_graph
# Wyświetla wizualny graf: węzły jako kółka, topics jako strzałki.
# Najszybszy sposób na wykrycie problemów z połączeniami.

# Pełna diagnostyka
rqt
# Uruchamia rqt – modularny interfejs graficzny z wtyczkami:
# - rqt_console: logi ze wszystkich węzłów
# - rqt_plot: wykresy wartości z topics
# - rqt_image_view: podgląd obrazów z kamer
# - rqt_tf_tree: drzewo transformacji TF2

# Wizualizacja 3D
rviz2
# Interaktywna wizualizacja:
# - chmury punktów LiDAR
# - obrazy z kamer
# - transformacje TF2 jako osie
# - plany ruchu manipulatora
# - mapa środowiska (SLAM)
```

## Debugowanie i Dobre Praktyki

### Typowe błędy i ich przyczyny

```bash
# Problem: węzły nie komunikują się mimo poprawnych nazw topics

# Krok 1: Sprawdź czy DDS widzi oba węzły w tej samej domenie
echo $ROS_DOMAIN_ID
# Jeśli pusty → domena 0 (domyślna). Oba procesy muszą mieć TĘ SAMĄ domenę.
# Różne domeny = izolacja sieci = brak komunikacji (celowe zachowanie).

# Krok 2: Sprawdź kompatybilność QoS
ros2 topic info /topic --verbose
# Wyświetla szczegółowe QoS publisherów i subskrybentów.
# Niezgodność reliability (RELIABLE vs BEST_EFFORT) = brak komunikacji.

# Krok 3: Sprawdź czy typ wiadomości się zgadza
ros2 topic list -t
# Jeśli publisher wysyła std_msgs/String a subscriber oczekuje geometry_msgs/Point
# → brak komunikacji. ROS2 nie zgłosi błędu – po prostu brak danych.
```

### Wzorzec: Executor i wielowątkowość

```python
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

class ComplexNode(Node):
    def __init__(self):
        super().__init__('complex_node')

        # MutuallyExclusiveCallbackGroup: callbacki w tej grupie nie mogą
        # działać równolegle (mutex). Domyślne zachowanie – bezpieczne.
        exclusive_group = MutuallyExclusiveCallbackGroup()

        # ReentrantCallbackGroup: callbacki mogą działać RÓWNOLEGLE.
        # Używamy gdy callback jest długotrwały (np. wywołanie modelu AI)
        # i nie chcemy blokować innych callbacków w tym czasie.
        reentrant_group = ReentrantCallbackGroup()

        self.sub1 = self.create_subscription(
            String, '/topic1', self.cb1, 10,
            callback_group=exclusive_group)

        self.sub2 = self.create_subscription(
            String, '/topic2', self.cb2, 10,
            callback_group=reentrant_group)  # może działać równolegle z cb1

    def cb1(self, msg):
        # Krótki callback – przetwarza dane i publikuje wynik
        pass

    def cb2(self, msg):
        # Długotrwały callback (np. inference modelu AI 100ms)
        # Dzięki ReentrantCallbackGroup nie blokuje cb1
        pass

def main():
    rclpy.init()
    node = ComplexNode()

    # MultiThreadedExecutor – obsługuje callbacki w wielu wątkach
    # num_threads=4 → do 4 callbacków jednocześnie
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
```

## Zasoby

- [Oficjalna Dokumentacja ROS2](https://docs.ros.org/en/humble/)
- [ROS2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [ROS2 Design](https://design.ros2.org/)
- [ROS Discourse](https://discourse.ros.org/)
- [ROS2 API (rclpy)](https://docs.ros2.org/latest/api/rclpy/)
- [DDS Standard (OMG)](https://www.omg.org/spec/DDS/)

## Następne Kroki

- [NVIDIA Isaac Lab](#wiki-isaac-lab) - symulacja robotów z ROS2
- [Moveit2](#wiki-motion-planning) - planowanie ruchu
- [Navigation2](https://navigation.ros.org/) - autonomiczna nawigacja

---

*Ostatnia aktualizacja: 2025-03-17*  
*Autor: Laboratorium Robotów Humanoidalnych, PRz*
