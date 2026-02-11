# Unitree G1 - Robot Humanoidalny

## Specyfikacja Techniczna

**Unitree G1** to zaawansowany robot humanoidalny wykorzystywany w Laboratorium Robotów Humanoidalnych do badań nad percepcją, kognicją i manipulacją.

### Parametry Fizyczne

| Parametr | Wartość |
|----------|---------|
| **Wysokość** | 127 cm |
| **Waga** | 35 kg |
| **DOF (Stopnie Swobody)** | 23 + 12 (ręce) |
| **Prędkość chodzenia** | Do 2 m/s |
| **Udźwig ręki** | 3 kg każda |
| **Czas pracy** | 2-4 godziny |
| **Bateria** | 15000 mAh |

## Układ Kinematyczny

### Nogi (12 DOF)

```python
# Konfiguracja stawów nóg
LEG_JOINTS = {
    'left_leg': [
        'left_hip_yaw',      # Biodro - rotacja
        'left_hip_roll',     # Biodro - abdukcja
        'left_hip_pitch',    # Biodro - zgięcie
        'left_knee',         # Kolano
        'left_ankle_pitch',  # Kostka - zgięcie
        'left_ankle_roll'    # Kostka - rotacja
    ],
    'right_leg': [
        'right_hip_yaw',
        'right_hip_roll',
        'right_hip_pitch',
        'right_knee',
        'right_ankle_pitch',
        'right_ankle_roll'
    ]
}

# Zakresy ruchu (stopnie)
JOINT_LIMITS = {
    'hip_yaw': (-45, 45),
    'hip_roll': (-30, 30),
    'hip_pitch': (-120, 60),
    'knee': (0, 150),
    'ankle_pitch': (-45, 45),
    'ankle_roll': (-20, 20)
}
```

### Górna Część Ciała (11 DOF)

```python
UPPER_BODY_JOINTS = {
    'torso': [
        'waist_yaw',     # Talia - rotacja
        'waist_pitch',   # Talia - przechył
        'waist_roll'     # Talia - pochylenie boczne
    ],
    'head': [
        'neck_yaw',      # Szyja - rotacja
        'neck_pitch'     # Szyja - przechył
    ],
    'arms': [
        'left_shoulder_pitch',
        'left_shoulder_roll',
        'left_elbow',
        'right_shoulder_pitch',
        'right_shoulder_roll',
        'right_elbow'
    ]
}
```

### Dłonie (12 DOF całkowite)

```python
HAND_JOINTS = {
    'left_hand': [
        'left_wrist_yaw',
        'left_wrist_pitch',
        'left_thumb',
        'left_index',
        'left_middle',
        'left_fingers'  # Pozostałe palce synergicznie
    ],
    'right_hand': [
        'right_wrist_yaw',
        'right_wrist_pitch',
        'right_thumb',
        'right_index',
        'right_middle',
        'right_fingers'
    ]
}
```

## System Sensoryczny

### Wizja

```python
CAMERA_SPECS = {
    'head_camera': {
        'type': 'RealSense D435i',
        'resolution': (1280, 720),
        'fps': 30,
        'fov': 87,  # stopni
        'depth_range': (0.3, 10.0),  # metry
        'imu': True  # Wbudowany IMU
    },
    'chest_camera': {
        'type': 'RGB',
        'resolution': (1920, 1080),
        'fps': 60,
        'fov': 110
    }
}
```

### IMU i Propriocepcja

```python
SENSOR_SUITE = {
    'imu': {
        'location': 'torso',
        'rate': 200,  # Hz
        'sensors': ['gyroscope', 'accelerometer', 'magnetometer']
    },
    'joint_encoders': {
        'resolution': 16384,  # pulses/revolution
        'accuracy': 0.02  # degrees
    },
    'force_sensors': {
        'feet': {
            'type': '6-axis force/torque',
            'range': (0, 500),  # N
            'location': ['left_foot', 'right_foot']
        },
        'wrists': {
            'type': '6-axis force/torque',
            'range': (0, 100),  # N
            'location': ['left_wrist', 'right_wrist']
        }
    },
    'tactile_sensors': {
        'hands': {
            'type': 'Pressure array',
            'resolution': '16x16',
            'sampling_rate': 100  # Hz
        }
    }
}
```

## Kontrola i Komunikacja

### ROS2 Interface

```python
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import JointState, Image, Imu
from std_msgs.msg import Float64MultiArray

class UnitreeG1Controller:
    def __init__(self):
        # Publishers
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/unitree_g1/joint_commands',
            10
        )
        
        self.velocity_pub = self.create_publisher(
            Twist,
            '/unitree_g1/cmd_vel',
            10
        )
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/unitree_g1/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/unitree_g1/imu',
            self.imu_callback,
            10
        )
    
    def set_joint_positions(self, positions):
        """
        Ustaw pozycje wszystkich stawów
        
        Args:
            positions: dict {joint_name: angle_rad}
        """
        msg = Float64MultiArray()
        msg.data = [positions[joint] for joint in JOINT_ORDER]
        self.joint_cmd_pub.publish(msg)
    
    def walk(self, vx, vy, vyaw):
        """
        Chodzenie
        
        Args:
            vx: prędkość do przodu (m/s)
            vy: prędkość w bok (m/s)
            vyaw: prędkość obrotowa (rad/s)
        """
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = vyaw
        
        self.velocity_pub.publish(msg)
```

### Kinematyka Prosta i Odwrotna

```python
import numpy as np
from scipy.spatial.transform import Rotation

class G1Kinematics:
    def __init__(self):
        # DH parameters dla nogi
        self.leg_dh = {
            'a': [0, 0, 0.35, 0.35, 0, 0],
            'd': [0.1, 0, 0, 0, 0, 0.05],
            'alpha': [np.pi/2, -np.pi/2, 0, 0, -np.pi/2, 0]
        }
    
    def forward_kinematics(self, joint_angles):
        """
        Kinematyka prosta - oblicz pozycję stopy
        
        Args:
            joint_angles: [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]
        
        Returns:
            4x4 transformation matrix
        """
        T = np.eye(4)
        
        for i, theta in enumerate(joint_angles):
            # DH transformation
            a = self.leg_dh['a'][i]
            d = self.leg_dh['d'][i]
            alpha = self.leg_dh['alpha'][i]
            
            T_i = np.array([
                [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]
            ])
            
            T = T @ T_i
        
        return T
    
    def inverse_kinematics(self, target_pose):
        """
        Kinematyka odwrotna - oblicz kąty stawów dla docelowej pozycji
        
        Args:
            target_pose: 4x4 transformation matrix or [x, y, z, roll, pitch, yaw]
        
        Returns:
            joint_angles: [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]
        """
        if target_pose.shape == (6,):
            # Convert to transformation matrix
            x, y, z, roll, pitch, yaw = target_pose
            R = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
        else:
            T = target_pose
        
        # Numerical IK (simplified)
        from scipy.optimize import minimize
        
        def cost(q):
            T_fk = self.forward_kinematics(q)
            pos_error = np.linalg.norm(T_fk[:3, 3] - T[:3, 3])
            rot_error = np.linalg.norm(T_fk[:3, :3] - T[:3, :3])
            return pos_error + 0.1 * rot_error
        
        # Initial guess
        q0 = np.array([0, 0, -0.5, 1.0, -0.5, 0])
        
        result = minimize(cost, q0, method='SLSQP')
        
        return result.x
```

## Locomotion - Generowanie Chodu

```python
class GaitGenerator:
    def __init__(self):
        self.step_height = 0.05  # m
        self.step_length = 0.2   # m
        self.step_width = 0.15   # m
        self.step_time = 0.4     # s
    
    def generate_walking_trajectory(self, vx, vy, vyaw, t):
        """
        Generuj trajektorię chodzenia
        
        Returns:
            left_foot_pose, right_foot_pose
        """
        # Faza kroku (0-1)
        phase = (t / self.step_time) % 1.0
        
        # Foot positions
        if phase < 0.5:
            # Left foot swing
            swing_foot = 'left'
            progress = phase * 2
        else:
            # Right foot swing
            swing_foot = 'right'
            progress = (phase - 0.5) * 2
        
        # Swing foot trajectory
        swing_x = self.step_length * progress
        swing_y = self.step_width/2 if swing_foot == 'left' else -self.step_width/2
        swing_z = self.step_height * np.sin(np.pi * progress)
        
        # Create poses
        if swing_foot == 'left':
            left_pose = [swing_x, swing_y, swing_z, 0, 0, 0]
            right_pose = [0, -self.step_width/2, 0, 0, 0, 0]
        else:
            left_pose = [0, self.step_width/2, 0, 0, 0, 0]
            right_pose = [swing_x, swing_y, swing_z, 0, 0, 0]
        
        return left_pose, right_pose
```

## Przykład Użycia - Podstawowa Kontrola

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class G1WalkingDemo(Node):
    def __init__(self):
        super().__init__('g1_walking_demo')
        
        self.controller = UnitreeG1Controller()
        self.kinematics = G1Kinematics()
        self.gait = GaitGenerator()
        
        # Control loop timer
        self.timer = self.create_timer(0.01, self.control_loop)
        self.t = 0.0
    
    def control_loop(self):
        """
        Main control loop - 100 Hz
        """
        # Generate walking trajectory
        left_pose, right_pose = self.gait.generate_walking_trajectory(
            vx=0.3, vy=0.0, vyaw=0.0, t=self.t
        )
        
        # Inverse kinematics
        left_joints = self.kinematics.inverse_kinematics(left_pose)
        right_joints = self.kinematics.inverse_kinematics(right_pose)
        
        # Send commands
        joint_positions = {}
        for i, joint in enumerate(LEG_JOINTS['left_leg']):
            joint_positions[joint] = left_joints[i]
        for i, joint in enumerate(LEG_JOINTS['right_leg']):
            joint_positions[joint] = right_joints[i]
        
        self.controller.set_joint_positions(joint_positions)
        
        self.t += 0.01

def main():
    rclpy.init()
    node = G1WalkingDemo()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

## Bezpieczeństwo

### Limity i Monitorowanie

```python
class SafetyMonitor:
    def __init__(self):
        self.max_joint_velocity = 3.0  # rad/s
        self.max_torque = 50.0  # Nm
        self.min_battery = 20.0  # %
        
        self.emergency_stop = False
    
    def check_safety(self, state):
        """
        Sprawdź warunki bezpieczeństwa
        """
        # Check joint velocities
        if np.any(np.abs(state.joint_velocities) > self.max_joint_velocity):
            self.trigger_emergency_stop("Joint velocity limit exceeded")
            return False
        
        # Check torques
        if np.any(np.abs(state.joint_torques) > self.max_torque):
            self.trigger_emergency_stop("Torque limit exceeded")
            return False
        
        # Check battery
        if state.battery_level < self.min_battery:
            self.get_logger().warn("Low battery!")
        
        # Check IMU (falling detection)
        if np.abs(state.imu.orientation.x) > 0.5:
            self.trigger_emergency_stop("Robot falling")
            return False
        
        return True
    
    def trigger_emergency_stop(self, reason):
        """
        Zatrzymanie awaryjne
        """
        self.get_logger().error(f"EMERGENCY STOP: {reason}")
        self.emergency_stop = True
        # Send zero velocities
        # Enable servo brakes
```

## Powiązane Artykuły

- [ROS2](#wiki-ros2)
- [Kinematics](#wiki-kinematics)
- [Motion Planning](#wiki-motion-planning)
- [Isaac Lab](#wiki-isaac-lab) - Symulacja G1

## Dokumentacja

- [Unitree Robotics](https://www.unitree.com/)
- [G1 SDK](https://github.com/unitreerobotics/unitree_sdk2)

---

*Ostatnia aktualizacja: 2025-02-11*  
*Autor: Zespół Robotyki, Laboratorium Robotów Humanoidalnych*
