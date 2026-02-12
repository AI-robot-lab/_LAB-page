# Bezpieczeństwo w Robotyce (Safety)

## Wprowadzenie

**Safety** to kluczowy aspekt robotyki humanoidalnej, obejmujący zapobieganie kolizjom, monitorowanie stanu i procedury awaryjne.

## Poziomy Bezpieczeństwa

### Safety System Architecture

```python
class SafetySystem:
    def __init__(self):
        self.emergency_stop = False
        self.safety_level = "NORMAL"  # NORMAL, WARNING, DANGER, EMERGENCY
        
        self.monitors = {
            'collision': CollisionMonitor(),
            'joint_limits': JointLimitsMonitor(),
            'temperature': TemperatureMonitor(),
            'power': PowerMonitor()
        }
    
    def check_safety(self, robot_state):
        """
        Comprehensive safety check
        """
        # Check all monitors
        for name, monitor in self.monitors.items():
            status = monitor.check(robot_state)
            
            if status == "EMERGENCY":
                self.trigger_emergency_stop()
                return "EMERGENCY"
            elif status == "DANGER":
                self.safety_level = "DANGER"
            elif status == "WARNING" and self.safety_level == "NORMAL":
                self.safety_level = "WARNING"
        
        return self.safety_level
    
    def trigger_emergency_stop(self):
        """
        Emergency stop procedure
        """
        self.emergency_stop = True
        self.safety_level = "EMERGENCY"
        
        # Stop all motors
        self.stop_all_actuators()
        
        # Enable brakes
        self.engage_brakes()
        
        # Log event
        self.log_emergency("Emergency stop triggered")
```

## Collision Detection

### Virtual Collision Zones

```python
class CollisionMonitor:
    def __init__(self):
        self.collision_zones = {
            'head': {'radius': 0.15, 'priority': 'HIGH'},
            'torso': {'radius': 0.25, 'priority': 'MEDIUM'},
            'arms': {'radius': 0.10, 'priority': 'LOW'}
        }
    
    def check_collision(self, robot_parts, obstacles):
        """
        Check for potential collisions
        """
        for part_name, part_pos in robot_parts.items():
            zone = self.collision_zones.get(part_name)
            
            if not zone:
                continue
            
            for obstacle in obstacles:
                distance = np.linalg.norm(part_pos - obstacle['position'])
                
                if distance < (zone['radius'] + obstacle['radius']):
                    return zone['priority']
        
        return "SAFE"
```

### Force/Torque Monitoring

```python
class ForceMonitor:
    def __init__(self):
        self.force_limits = {
            'contact_force': 50.0,  # N
            'joint_torque': 100.0   # Nm
        }
    
    def check(self, sensor_data):
        """
        Monitor force/torque sensors
        """
        # Check external force
        if np.linalg.norm(sensor_data['contact_force']) > self.force_limits['contact_force']:
            return "DANGER"
        
        # Check joint torques
        for joint, torque in sensor_data['joint_torques'].items():
            if abs(torque) > self.force_limits['joint_torque']:
                return "WARNING"
        
        return "SAFE"
```

## Joint Limits

```python
class JointLimitsMonitor:
    def __init__(self, limits):
        self.position_limits = limits['position']
        self.velocity_limits = limits['velocity']
        self.acceleration_limits = limits['acceleration']
    
    def check(self, robot_state):
        """
        Check if joints within safe limits
        """
        # Position limits
        for joint, pos in robot_state['positions'].items():
            low, high = self.position_limits[joint]
            
            if pos < low or pos > high:
                return "EMERGENCY"
            
            # Warning zone (10% margin)
            margin = 0.1 * (high - low)
            if pos < (low + margin) or pos > (high - margin):
                return "WARNING"
        
        # Velocity limits
        for joint, vel in robot_state['velocities'].items():
            if abs(vel) > self.velocity_limits[joint]:
                return "DANGER"
        
        return "SAFE"
```

## Watchdog Timer

```python
import threading
import time

class WatchdogTimer:
    def __init__(self, timeout=1.0):
        self.timeout = timeout
        self.last_ping = time.time()
        self.running = True
        
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
    
    def ping(self):
        """
        Reset watchdog timer
        """
        self.last_ping = time.time()
    
    def _monitor(self):
        """
        Monitor thread
        """
        while self.running:
            if (time.time() - self.last_ping) > self.timeout:
                self.trigger_timeout()
            
            time.sleep(0.1)
    
    def trigger_timeout(self):
        """
        Handle timeout
        """
        print("WATCHDOG TIMEOUT - EMERGENCY STOP")
        # Trigger emergency stop
```

## ISO 13482 Compliance

```python
class ISO13482Compliance:
    """
    Safety requirements for personal care robots
    """
    
    @staticmethod
    def check_static_stability(robot):
        """
        Ensure robot is statically stable
        """
        com = robot.center_of_mass()
        support_polygon = robot.support_polygon()
        
        return point_in_polygon(com[:2], support_polygon)
    
    @staticmethod
    def check_emergency_stop():
        """
        Emergency stop must be easily accessible
        """
        # Implementation depends on hardware
        pass
    
    @staticmethod
    def check_pinch_points():
        """
        No pinch points accessible during operation
        """
        # Check joint gaps, moving parts
        pass
```

## Safety-Rated Monitoring

```python
class SafetyRatedController:
    def __init__(self):
        self.safe_mode = False
        self.performance_level = "PLd"  # ISO 13849-1
    
    def enable_safe_mode(self):
        """
        Enter reduced performance mode
        """
        self.safe_mode = True
        
        # Reduce velocities
        self.max_velocity = 0.25  # m/s
        self.max_joint_velocity = 0.5  # rad/s
        
        # Increase sensor polling
        self.sensor_rate = 100  # Hz
    
    def validate_trajectory(self, trajectory):
        """
        Validate trajectory meets safety constraints
        """
        for point in trajectory:
            # Check velocities
            if point.velocity > self.max_velocity:
                return False
            
            # Check accelerations
            if point.acceleration > self.max_acceleration:
                return False
        
        return True
```

## Powiązane Artykuły

- [HRI](#wiki-hri)
- [Control Theory](#wiki-control-theory)
- [Ethics](#wiki-ethics)

---

*Ostatnia aktualizacja: 2025-02-11*
