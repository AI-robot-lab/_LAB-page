# Manipulacja Robotyczna

## Wprowadzenie

**Manipulacja robotyczna** to zdolność robota do fizycznej interakcji z otoczeniem - chwytania, przemieszczania i manipulowania obiektami. W robotyce humanoidalnej wymaga precyzyjnej koordynacji percepcji, planowania i kontroli.

## Grasp Planning

### Grasp Types

```python
import numpy as np

class GraspPlanner:
    def __init__(self):
        # Typy chwytów
        self.grasp_types = {
            'power_grasp': {
                'description': 'Chwyt siłowy - cały obiekt w dłoni',
                'fingers': [1, 1, 1, 1, 1],  # Wszystkie palce
                'force': 'high'
            },
            'precision_grasp': {
                'description': 'Chwyt precyzyjny - kciuk + palce',
                'fingers': [1, 1, 0, 0, 0],  # Kciuk + wskazujący
                'force': 'low'
            },
            'pinch_grasp': {
                'description': 'Chwyt szczypcem - 2 palce',
                'fingers': [1, 1, 0, 0, 0],
                'force': 'medium'
            }
        }
    
    def plan_grasp(self, object_properties):
        """
        Planuj chwyt na podstawie właściwości obiektu
        """
        size = object_properties['size']
        weight = object_properties['weight']
        fragile = object_properties['fragile']
        
        # Wybierz typ chwytu
        if size < 0.05 and fragile:
            return 'precision_grasp'
        elif weight > 1.0:
            return 'power_grasp'
        else:
            return 'pinch_grasp'
    
    def compute_grasp_pose(self, object_pose, grasp_type):
        """
        Oblicz pozę dłoni dla chwytu
        """
        # Współrzędne obiektu
        obj_pos = object_pose[:3]
        obj_orient = object_pose[3:]
        
        # Offset dla różnych typów chwytów
        if grasp_type == 'power_grasp':
            approach_vector = np.array([0, 0, -1])  # Z góry
        elif grasp_type == 'precision_grasp':
            approach_vector = np.array([0, -1, 0])  # Z boku
        else:
            approach_vector = np.array([1, 0, 0])   # Z przodu
        
        # Oblicz pozę chwytu
        grasp_pos = obj_pos + approach_vector * 0.1
        grasp_orient = self.compute_grasp_orientation(approach_vector)
        
        return np.concatenate([grasp_pos, grasp_orient])
```

## Dexterous Manipulation

### Hand Control

```python
class DexterousHand:
    def __init__(self, num_fingers=5):
        self.num_fingers = num_fingers
        
        # Stopnie swobody dla każdego palca
        self.dof_per_finger = 3  # Podstawa, środek, czubek
        
        # Limity stawów (radiany)
        self.joint_limits = {
            'base': (0, np.pi/2),
            'middle': (0, np.pi/2),
            'tip': (0, np.pi/3)
        }
    
    def close_fingers(self, finger_indices, target_angles):
        """
        Zamknij wybrane palce
        
        Args:
            finger_indices: lista indeksów palców [0-4]
            target_angles: kąty dla każdego stawu
        """
        commands = {}
        
        for finger_id in finger_indices:
            for joint_id, angle in enumerate(target_angles):
                joint_name = f"finger_{finger_id}_joint_{joint_id}"
                
                # Sprawdź limity
                angle = np.clip(
                    angle,
                    self.joint_limits['base'][0],
                    self.joint_limits['base'][1]
                )
                
                commands[joint_name] = angle
        
        return commands
    
    def adaptive_grasp(self, contact_forces):
        """
        Adaptacyjny chwyt - dostosuj siłę na podstawie kontaktu
        """
        target_force = 5.0  # N
        
        # Dla każdego palca
        adjustments = {}
        for finger_id, force in contact_forces.items():
            error = target_force - force
            
            # Proporcjonalna korekta
            adjustment = 0.1 * error
            adjustments[finger_id] = adjustment
        
        return adjustments
```

## Impedance Control

```python
class ImpedanceController:
    def __init__(self, M_d, D_d, K_d):
        """
        Impedance controller dla manipulacji
        
        Args:
            M_d: Desired mass matrix
            D_d: Desired damping matrix
            K_d: Desired stiffness matrix
        """
        self.M_d = M_d
        self.D_d = D_d
        self.K_d = K_d
    
    def compute_force(self, x, x_d, v, v_d, f_ext):
        """
        Oblicz siłę sterującą
        
        M_d*(ẍ_d - ẍ) + D_d*(ẋ_d - ẋ) + K_d*(x_d - x) = f_ext
        """
        # Błędy
        e_pos = x_d - x
        e_vel = v_d - v
        
        # Siła sterująca
        f_control = self.K_d @ e_pos + self.D_d @ e_vel - f_ext
        
        return f_control
    
    def set_compliance(self, direction, stiffness):
        """
        Ustaw podatność w określonym kierunku
        """
        # Zmień stiffness w wybranym kierunku
        if direction == 'x':
            self.K_d[0, 0] = stiffness
        elif direction == 'y':
            self.K_d[1, 1] = stiffness
        elif direction == 'z':
            self.K_d[2, 2] = stiffness
```

## Visual Servoing

```python
class VisualServoing:
    def __init__(self, camera_matrix):
        self.K = camera_matrix
        self.lambda_gain = 0.5
    
    def image_jacobian(self, features_2d, depth):
        """
        Oblicz Image Jacobian
        
        Args:
            features_2d: punkty w obrazie (u, v)
            depth: głębokość punktów (Z)
        """
        u, v = features_2d
        Z = depth
        
        # Image Jacobian (2x6)
        L = np.array([
            [-1/Z, 0, u/Z, u*v, -(1+u**2), v],
            [0, -1/Z, v/Z, 1+v**2, -u*v, -u]
        ])
        
        return L
    
    def compute_velocity(self, current_features, desired_features, depth):
        """
        Oblicz prędkość end-effector
        """
        # Błąd w przestrzeni obrazu
        e = current_features - desired_features
        
        # Image Jacobian
        L = self.image_jacobian(current_features, depth)
        
        # Pseudo-inverse
        L_pinv = np.linalg.pinv(L)
        
        # Prędkość
        v = -self.lambda_gain * L_pinv @ e
        
        return v
```

## Force Control

### Hybrid Position/Force Control

```python
class HybridController:
    def __init__(self):
        # Selection matrices
        self.S_pos = np.eye(6)  # Position control directions
        self.S_force = np.zeros((6, 6))  # Force control directions
    
    def set_control_mode(self, force_directions):
        """
        Ustaw które kierunki są kontrolowane siłą
        
        Args:
            force_directions: lista kierunków [0-5]
                             0-2: translacja (x,y,z)
                             3-5: rotacja (rx,ry,rz)
        """
        self.S_pos = np.eye(6)
        self.S_force = np.zeros((6, 6))
        
        for dir in force_directions:
            self.S_pos[dir, dir] = 0
            self.S_force[dir, dir] = 1
    
    def control(self, x, x_d, f, f_d, K_p, K_f):
        """
        Hybrid position/force control
        
        Args:
            x, x_d: aktualna i pożądana pozycja
            f, f_d: aktualna i pożądana siła
            K_p, K_f: wzmocnienia
        """
        # Position error
        e_pos = x_d - x
        
        # Force error
        e_force = f_d - f
        
        # Control command
        u = self.S_pos @ (K_p @ e_pos) + self.S_force @ (K_f @ e_force)
        
        return u
```

## Learning-Based Manipulation

### Imitation Learning

```python
import torch
import torch.nn as nn

class ManipulationPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def forward(self, obs):
        return self.network(obs)

class BehavioralCloning:
    def __init__(self, policy):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    def train(self, demonstrations):
        """
        Ucz się z demonstracji
        
        Args:
            demonstrations: lista (observation, action)
        """
        for obs, action in demonstrations:
            # Predict
            predicted_action = self.policy(obs)
            
            # Loss
            loss = nn.functional.mse_loss(predicted_action, action)
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

## Contact-Rich Manipulation

```python
class ContactRichController:
    def __init__(self):
        self.contact_threshold = 1.0  # N
        self.slide_threshold = 0.1    # m/s
    
    def detect_contact(self, force_readings):
        """
        Wykryj kontakt na podstawie czujników siły
        """
        contact_detected = {}
        
        for sensor_name, force in force_readings.items():
            if np.linalg.norm(force) > self.contact_threshold:
                contact_detected[sensor_name] = True
            else:
                contact_detected[sensor_name] = False
        
        return contact_detected
    
    def adjust_for_contact(self, contact_state, desired_velocity):
        """
        Dostosuj ruch w przypadku kontaktu
        """
        if any(contact_state.values()):
            # Zmniejsz prędkość
            adjusted_velocity = desired_velocity * 0.5
            
            # Zwiększ compliance
            adjusted_stiffness = 100  # N/m (niskie)
        else:
            adjusted_velocity = desired_velocity
            adjusted_stiffness = 1000  # N/m (wysokie)
        
        return adjusted_velocity, adjusted_stiffness
```

## Powiązane Artykuły

- [Motion Planning](#wiki-motion-planning)
- [Kinematics](#wiki-kinematics)
- [Control Theory](#wiki-control-theory)
- [Reinforcement Learning](#wiki-reinforcement-learning)

---

*Ostatnia aktualizacja: 2025-02-12*  
*Autor: Zespół Interakcji, Laboratorium Robotów Humanoidalnych*
