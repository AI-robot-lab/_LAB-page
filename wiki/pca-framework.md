# Framework PCA (Perception-Cognition-Action)

## Wprowadzenie

**Framework PCA** to architektura przetwarzania informacji w autonomicznych systemach robotycznych, wykorzystywana w Laboratorium Robotów Humanoidalnych PRz. Składa się z trzech głównych modułów działających w zamkniętej pętli poznawczej.

```
┌─────────────────────────────────────────┐
│           PERCEPCJA (P)                 │
│   ┌──────────────────────────────┐     │
│   │ Sensory → Preprocessing →    │     │
│   │ Feature Extraction           │     │
│   └──────────────────────────────┘     │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│            KOGNICJA (C)                 │
│   ┌──────────────────────────────┐     │
│   │ World Model → Reasoning →    │     │
│   │ Decision Making              │     │
│   └──────────────────────────────┘     │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│              AKCJA (A)                  │
│   ┌──────────────────────────────┐     │
│   │ Planning → Control →         │     │
│   │ Execution                    │     │
│   └──────────────────────────────┘     │
└──────────────┬──────────────────────────┘
               ↓
        [Środowisko]
               ↓
        (Feedback Loop)
```

## Moduł Percepcji

### Zadania
- **Fuzja sensoryczna** - integracja danych z wielu źródeł
- **Detekcja obiektów** - identyfikacja elementów środowiska
- **Śledzenie** - monitorowanie zmian w czasie
- **Mapowanie** - reprezentacja przestrzenna

### Sensory w Unitree G1

| Sensor | Typ | Zastosowanie |
|--------|-----|--------------|
| LiDAR MID360 | 3D Point Cloud | Mapowanie, obstacle detection |
| RGB Cameras | Wizja stereoskopowa | Detekcja, tracking |
| Depth Camera | Intel RealSense | Estymacja głębi |
| IMU | 9-DOF | Orientacja, akceleracja |
| Mikrofony | 8-kanałowy array | Lokalizacja źródła dźwięku |

### Pipeline Percepcji

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseStamped

class PerceptionModule(Node):
    def __init__(self):
        super().__init__('perception_module')
        
        # Subskrypcje
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/lidar/points', self.lidar_callback, 10
        )
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10
        )
        
        # Publikacje
        self.objects_pub = self.create_publisher(
            DetectedObjects, '/perception/objects', 10
        )
        self.map_pub = self.create_publisher(
            OccupancyGrid, '/perception/map', 10
        )
    
    def lidar_callback(self, msg):
        # Przetwarzanie chmury punktów
        points = self.pointcloud_to_array(msg)
        
        # Segmentacja
        clusters = self.cluster_points(points)
        
        # Detekcja obiektów
        objects = self.detect_objects(clusters)
        
        # Publikacja
        self.objects_pub.publish(objects)
    
    def camera_callback(self, msg):
        # Konwersja do numpy
        image = self.imgmsg_to_cv2(msg)
        
        # Detekcja twarzy i emocji
        faces = self.face_detector.detect(image)
        emotions = self.emotion_recognizer.analyze(faces)
        
        # Publikacja wyników
        self.publish_affective_state(emotions)
```

## Moduł Kognicji

### Zadania
- **Modelowanie świata** - reprezentacja środowiska
- **Wnioskowanie** - dedukacja na podstawie danych
- **Planowanie** - strategia działania
- **Uczenie się** - adaptacja na podstawie doświadczeń

### Architektura Kognicji

```python
class CognitionModule(Node):
    def __init__(self):
        super().__init__('cognition_module')
        
        # World Model
        self.world_model = WorldModel()
        
        # LLM dla reasoning
        self.llm = LargeLanguageModel(model="claude-3-sonnet")
        
        # VLM dla vision-language tasks
        self.vlm = VisionLanguageModel()
        
        # Planner
        self.planner = MotionPlanner()
        
        # Subskrypcje z percepcji
        self.perception_sub = self.create_subscription(
            PerceptionData, '/perception/data', 
            self.perception_callback, 10
        )
        
        # Publikacje do akcji
        self.action_pub = self.create_publisher(
            ActionCommand, '/cognition/action', 10
        )
    
    def perception_callback(self, msg):
        # Aktualizacja world model
        self.world_model.update(msg)
        
        # Reasoning
        context = self.world_model.get_context()
        decision = self.llm.reason(context)
        
        # Planning
        if decision.requires_motion:
            plan = self.planner.plan_trajectory(
                start=self.world_model.robot_pose,
                goal=decision.target_pose,
                obstacles=self.world_model.obstacles
            )
            
            # Publikacja akcji
            self.action_pub.publish(
                ActionCommand(type='execute_plan', plan=plan)
            )
```

### Przykład: Rozpoznawanie Intencji

```python
class IntentRecognition:
    def __init__(self):
        self.llm = initialize_llm()
        self.context_window = []
    
    def recognize_intent(self, user_input, visual_context):
        # Kontekst
        self.context_window.append({
            'user_input': user_input,
            'visual': visual_context,
            'timestamp': time.time()
        })
        
        # Prompt dla LLM
        prompt = f"""
        Kontekst: {self.context_window[-5:]}
        Użytkownik: {user_input}
        Scena wizualna: {visual_context}
        
        Rozpoznaj intencję użytkownika i zaproponuj akcję robota.
        Odpowiedź w formacie JSON:
        {{
            "intent": "string",
            "confidence": 0-1,
            "proposed_action": "string",
            "requires_clarification": boolean
        }}
        """
        
        response = self.llm.generate(prompt)
        return json.loads(response)
```

## Moduł Akcji

### Zadania
- **Planowanie trajektorii** - optymalne ścieżki ruchu
- **Sterowanie** - wykonanie ruchu
- **Manipulacja** - interakcja z obiektami
- **Stabilizacja** - utrzymanie równowagi

### Pipeline Akcji

```python
class ActionModule(Node):
    def __init__(self):
        super().__init__('action_module')
        
        # Kontroler manipulatora
        self.manipulator = ManipulatorController()
        
        # Kontroler lokomotion
        self.locomotion = LocomotionController()
        
        # MoveIt2 dla motion planning
        self.moveit = MoveItInterface()
        
        # Subskrypcja z kognicji
        self.cognition_sub = self.create_subscription(
            ActionCommand, '/cognition/action',
            self.action_callback, 10
        )
        
        # Publisher statusu
        self.status_pub = self.create_publisher(
            ActionStatus, '/action/status', 10
        )
    
    def action_callback(self, msg):
        if msg.type == 'grasp_object':
            self.execute_grasp(msg.target)
        
        elif msg.type == 'navigate_to':
            self.execute_navigation(msg.goal_pose)
        
        elif msg.type == 'gesture':
            self.execute_gesture(msg.gesture_type)
    
    def execute_grasp(self, target):
        # Planowanie IK
        joint_trajectory = self.moveit.plan_to_pose(
            target.pose,
            end_effector='right_hand'
        )
        
        # Wykonanie
        success = self.manipulator.execute_trajectory(joint_trajectory)
        
        # Zamknięcie gripper
        if success:
            self.manipulator.close_gripper()
        
        # Status
        self.status_pub.publish(
            ActionStatus(
                action='grasp',
                success=success,
                timestamp=self.get_clock().now()
            )
        )
```

## Integracja PCA w ROS2

### Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Percepcja
        Node(
            package='humanoid_perception',
            executable='lidar_processor',
            name='lidar_processor'
        ),
        Node(
            package='humanoid_perception',
            executable='vision_processor',
            name='vision_processor'
        ),
        Node(
            package='humanoid_perception',
            executable='affective_analyzer',
            name='affective_analyzer'
        ),
        
        # Kognicja
        Node(
            package='humanoid_cognition',
            executable='world_model',
            name='world_model'
        ),
        Node(
            package='humanoid_cognition',
            executable='reasoning_engine',
            name='reasoning_engine'
        ),
        
        # Akcja
        Node(
            package='humanoid_action',
            executable='motion_controller',
            name='motion_controller'
        ),
        Node(
            package='humanoid_action',
            executable='manipulator_controller',
            name='manipulator_controller'
        ),
    ])
```

## Zamknięta Pętla Poznawcza

### Feedback Loop

```python
class PCAFeedbackLoop:
    def __init__(self):
        self.perception = PerceptionModule()
        self.cognition = CognitionModule()
        self.action = ActionModule()
        
        # Performance metrics
        self.metrics = {
            'perception_latency': [],
            'cognition_latency': [],
            'action_latency': [],
            'success_rate': []
        }
    
    def run_cycle(self):
        # 1. Percepcja
        t0 = time.time()
        perception_data = self.perception.sense_environment()
        t1 = time.time()
        
        # 2. Kognicja
        decision = self.cognition.process(perception_data)
        t2 = time.time()
        
        # 3. Akcja
        result = self.action.execute(decision)
        t3 = time.time()
        
        # 4. Ewaluacja
        success = self.evaluate_outcome(result, decision.goal)
        
        # 5. Uczenie się
        if not success:
            self.cognition.update_policy(
                state=perception_data,
                action=decision,
                reward=-1.0
            )
        
        # Metryki
        self.metrics['perception_latency'].append(t1 - t0)
        self.metrics['cognition_latency'].append(t2 - t1)
        self.metrics['action_latency'].append(t3 - t2)
        self.metrics['success_rate'].append(float(success))
```

## Optymalizacja Framework

### Latency Reduction

```python
# Asynchroniczne przetwarzanie
import asyncio

class AsyncPCAFramework:
    async def perception_pipeline(self):
        while True:
            data = await self.capture_sensors()
            processed = await self.process_perception(data)
            await self.perception_queue.put(processed)
    
    async def cognition_pipeline(self):
        while True:
            perception_data = await self.perception_queue.get()
            decision = await self.make_decision(perception_data)
            await self.action_queue.put(decision)
    
    async def action_pipeline(self):
        while True:
            decision = await self.action_queue.get()
            result = await self.execute_action(decision)
            await self.feedback_queue.put(result)
    
    async def run(self):
        await asyncio.gather(
            self.perception_pipeline(),
            self.cognition_pipeline(),
            self.action_pipeline()
        )
```

## Zastosowania

### Rehabilitacja Wspomagana

- **Percepcja**: Analiza ruchu pacjenta, detekcja błędów
- **Kognicja**: Ocena postępu, dostosowanie trudności
- **Akcja**: Assist-as-needed, korekcja ruchu

### Interakcja Społeczna

- **Percepcja**: Rozpoznawanie emocji, intencji
- **Kognicja**: Teoria umysłu, empatie
- **Akcja**: Gesty, ekspresje, mowa

## Powiązane Artykuły

- [ROS2](#wiki-ros2) - implementacja komunikacji
- [Isaac Lab](#wiki-isaac-lab) - symulacja PCA
- [Informatyka Afektywna](#wiki-affective-computing) - moduł percepcji

---

*Ostatnia aktualizacja: 2025-02-10*  
*Autor: Laboratorium Robotów Humanoidalnych, PRz*
