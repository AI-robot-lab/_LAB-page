# ROS2 - Robot Operating System

## Wprowadzenie

**ROS2** (Robot Operating System 2) to otwarte oprogramowanie middleware do budowy aplikacji robotycznych. Jest to druga generacja frameworka ROS, zaprojektowana z myślą o systemach robotycznych działających w czasie rzeczywistym oraz środowiskach produkcyjnych.

## Architektura ROS2

### DDS Middleware

ROS2 bazuje na standardzie **DDS** (Data Distribution Service), który zapewnia:
- Komunikację peer-to-peer bez centralnego brokera
- Quality of Service (QoS) profiles
- Descobering automatyczne węzłów
- Real-time performance

### Główne Komponenty

1. **Nodes (węzły)** - podstawowe jednostki wykonawcze
2. **Topics** - kanały komunikacji asynchronicznej
3. **Services** - komunikacja synchroniczna request-response
4. **Actions** - długotrwałe zadania z feedbackiem
5. **Parameters** - dynamiczna konfiguracja

## Instalacja ROS2 Humble

```bash
# Ubuntu 22.04 LTS
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Dodaj repozytorium ROS2
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Dodaj źródło
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Instalacja
sudo apt update
sudo apt install ros-humble-desktop
```

## Podstawowy Przykład

### Publisher Node (Python)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = MinimalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### Subscriber Node (Python)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Quality of Service (QoS)

ROS2 pozwala na precyzyjną konfigurację QoS dla każdego topic:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,  # lub BEST_EFFORT
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

self.publisher_ = self.create_publisher(
    String, 
    'topic', 
    qos_profile
)
```

## Launch Files

Launch files pozwalają na uruchamianie wielu węzłów jednocześnie:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='publisher',
            name='publisher_node'
        ),
        Node(
            package='my_package',
            executable='subscriber',
            name='subscriber_node'
        ),
    ])
```

## Użycie w Laboratorium

W naszym laboratorium ROS2 Humble jest podstawowym frameworkiem do:
- Komunikacji między modułami robota Unitree G1
- Integracji sensorów (LiDAR, kamery, mikrofony)
- Sterowania manipulatorami Dex3-1
- Integracji z systemami AI (PyTorch, TensorFlow)

## Narzędzia CLI

```bash
# Lista węzłów
ros2 node list

# Informacje o węźle
ros2 node info /node_name

# Lista topics
ros2 topic list

# Echo topic
ros2 topic echo /topic_name

# Publikowanie wiadomości
ros2 topic pub /topic_name std_msgs/String "data: 'Hello'"

# Lista serwisów
ros2 service list

# Wywołanie serwisu
ros2 service call /service_name std_srvs/Trigger
```

## Zasoby

- [Oficjalna Dokumentacja ROS2](https://docs.ros.org/en/humble/)
- [ROS2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [ROS2 Design](https://design.ros2.org/)
- [ROS Discourse](https://discourse.ros.org/)

## Następne Kroki

- [NVIDIA Isaac Lab](#wiki-isaac-lab) - symulacja robotów z ROS2
- [Moveit2](#wiki-motion-planning) - planowanie ruchu
- [Navigation2](https://navigation.ros.org/) - autonomiczna nawigacja

---

*Ostatnia aktualizacja: 2025-02-10*  
*Autor: Laboratorium Robotów Humanoidalnych, PRz*
