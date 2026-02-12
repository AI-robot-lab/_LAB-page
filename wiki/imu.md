# IMU - Inertial Measurement Unit

## Wprowadzenie

**IMU (Inertial Measurement Unit)** to układ pomiarowy zawierający czujniki inercjalne: akcelerometr, żyroskop i opcjonalnie magnetometr. W robotyce humanoidalnej IMU jest kluczowy dla utrzymania równowagi i estymacji pozycji.

## Składowe IMU

### 3-osiowy Akcelerometr

```python
import numpy as np

class Accelerometer:
    def __init__(self):
        self.gravity = 9.81  # m/s^2
        
        # Noise parameters
        self.noise_density = 0.002  # g/√Hz
        self.bias = np.random.randn(3) * 0.01
    
    def measure(self, true_acceleration, orientation):
        """
        Pomiar przyspieszenia w ramce czujnika
        
        Args:
            true_acceleration: [ax, ay, az] w ramce globalnej
            orientation: kąty Eulera [roll, pitch, yaw]
        """
        # Dodaj grawitację
        g_global = np.array([0, 0, -self.gravity])
        
        # Transformuj do ramki czujnika
        R = self.euler_to_rotation_matrix(orientation)
        g_sensor = R.T @ g_global
        
        # Zmierzone przyspieszenie
        measured = true_acceleration + g_sensor
        
        # Dodaj szum i bias
        noise = np.random.randn(3) * self.noise_density
        measured = measured + noise + self.bias
        
        return measured
    
    def euler_to_rotation_matrix(self, euler):
        """
        Konwersja kątów Eulera na macierz rotacji
        """
        roll, pitch, yaw = euler
        
        # Roll (X)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Pitch (Y)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Yaw (Z)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx
```

### 3-osiowy Żyroskop

```python
class Gyroscope:
    def __init__(self):
        # Noise parameters
        self.noise_density = 0.001  # rad/s/√Hz
        self.bias = np.random.randn(3) * 0.01
        self.bias_drift = 0.0001  # rad/s per second
    
    def measure(self, true_angular_velocity, dt=0.01):
        """
        Pomiar prędkości kątowej
        
        Args:
            true_angular_velocity: [wx, wy, wz] w rad/s
            dt: czas od ostatniego pomiaru
        """
        # Bias drift
        self.bias += np.random.randn(3) * self.bias_drift * dt
        
        # Szum
        noise = np.random.randn(3) * self.noise_density
        
        # Pomiar
        measured = true_angular_velocity + noise + self.bias
        
        return measured
```

### 3-osiowy Magnetometr

```python
class Magnetometer:
    def __init__(self):
        # Pole magnetyczne Ziemi (przykład dla Polski)
        self.earth_field = np.array([0.2, 0.0, 0.45])  # Gauss
        
        # Hard iron offset (stałe zakłócenia)
        self.hard_iron = np.random.randn(3) * 0.1
        
        # Soft iron (zależne od orientacji)
        self.soft_iron = np.eye(3) + np.random.randn(3, 3) * 0.05
    
    def measure(self, orientation):
        """
        Pomiar pola magnetycznego
        
        Args:
            orientation: kąty Eulera [roll, pitch, yaw]
        """
        # Transformuj pole Ziemi do ramki czujnika
        R = self.euler_to_rotation_matrix(orientation)
        mag_sensor = R.T @ self.earth_field
        
        # Zastosuj zakłócenia
        mag_disturbed = self.soft_iron @ mag_sensor + self.hard_iron
        
        # Szum
        noise = np.random.randn(3) * 0.01
        measured = mag_disturbed + noise
        
        return measured
```

## Kompletny IMU Sensor

```python
class IMU:
    def __init__(self, sample_rate=100):
        self.accelerometer = Accelerometer()
        self.gyroscope = Gyroscope()
        self.magnetometer = Magnetometer()
        
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
    
    def read(self, true_state):
        """
        Odczyt wszystkich czujników
        
        Args:
            true_state: dict z kluczami:
                - acceleration: [ax, ay, az]
                - angular_velocity: [wx, wy, wz]
                - orientation: [roll, pitch, yaw]
        """
        accel = self.accelerometer.measure(
            true_state['acceleration'],
            true_state['orientation']
        )
        
        gyro = self.gyroscope.measure(
            true_state['angular_velocity'],
            self.dt
        )
        
        mag = self.magnetometer.measure(
            true_state['orientation']
        )
        
        return {
            'acceleration': accel,
            'angular_velocity': gyro,
            'magnetic_field': mag,
            'timestamp': time.time()
        }
```

## Kalibracja IMU

### Kalibracja Akcelerometru

```python
class IMUCalibration:
    def __init__(self):
        self.accel_bias = np.zeros(3)
        self.accel_scale = np.ones(3)
        
        self.gyro_bias = np.zeros(3)
        
        self.mag_hard_iron = np.zeros(3)
        self.mag_soft_iron = np.eye(3)
    
    def calibrate_accelerometer(self, measurements):
        """
        6-pozycyjna kalibracja akcelerometru
        
        Pozycje: +X, -X, +Y, -Y, +Z, -Z w górę
        """
        assert len(measurements) == 6
        
        g = 9.81
        
        # Ekstraktuj pomiary dla każdej osi
        pos_x = measurements[0]  # +X w górę
        neg_x = measurements[1]  # -X w górę
        pos_y = measurements[2]
        neg_y = measurements[3]
        pos_z = measurements[4]
        neg_z = measurements[5]
        
        # Oblicz bias
        self.accel_bias[0] = (pos_x[0] + neg_x[0]) / 2
        self.accel_bias[1] = (pos_y[1] + neg_y[1]) / 2
        self.accel_bias[2] = (pos_z[2] + neg_z[2]) / 2
        
        # Oblicz scale
        self.accel_scale[0] = 2 * g / (pos_x[0] - neg_x[0])
        self.accel_scale[1] = 2 * g / (pos_y[1] - neg_y[1])
        self.accel_scale[2] = 2 * g / (pos_z[2] - neg_z[2])
        
        print(f"Akcelerometr zkalibrowany:")
        print(f"  Bias: {self.accel_bias}")
        print(f"  Scale: {self.accel_scale}")
    
    def calibrate_gyroscope(self, measurements, duration=10):
        """
        Kalibracja żyroskopu (robot nieruchomy)
        
        Args:
            measurements: lista pomiarów prędkości kątowej
            duration: czas kalibracji w sekundach
        """
        # Średnia = bias
        self.gyro_bias = np.mean(measurements, axis=0)
        
        print(f"Żyroskop zkalibrowany:")
        print(f"  Bias: {self.gyro_bias}")
    
    def calibrate_magnetometer(self, measurements):
        """
        Kalibracja magnetometru (ruch w kształcie '8')
        
        Ellipsoid fitting dla hard/soft iron compensation
        """
        # Convert to numpy array
        data = np.array(measurements)
        
        # Fit ellipsoid
        center, radii, rotation = self.fit_ellipsoid(data)
        
        self.mag_hard_iron = center
        self.mag_soft_iron = rotation @ np.diag(1.0 / radii) @ rotation.T
        
        print(f"Magnetometr zkalibrowany:")
        print(f"  Hard iron: {self.mag_hard_iron}")
    
    def apply_calibration(self, raw_data):
        """
        Zastosuj kalibrację do surowych danych
        """
        # Akcelerometr
        accel_calibrated = (raw_data['acceleration'] - self.accel_bias) * self.accel_scale
        
        # Żyroskop
        gyro_calibrated = raw_data['angular_velocity'] - self.gyro_bias
        
        # Magnetometr
        mag_corrected = raw_data['magnetic_field'] - self.mag_hard_iron
        mag_calibrated = self.mag_soft_iron @ mag_corrected
        
        return {
            'acceleration': accel_calibrated,
            'angular_velocity': gyro_calibrated,
            'magnetic_field': mag_calibrated
        }
```

## Estymacja Orientacji

### Complementary Filter

```python
class ComplementaryFilter:
    def __init__(self, alpha=0.98):
        self.alpha = alpha  # Weight dla żyroskopu
        self.orientation = np.array([0.0, 0.0, 0.0])  # [roll, pitch, yaw]
    
    def update(self, accel, gyro, mag=None, dt=0.01):
        """
        Fuzja danych z IMU
        
        orientation = α*(orientation + gyro*dt) + (1-α)*accel_orientation
        """
        # Integracja żyroskopu
        gyro_orientation = self.orientation + gyro * dt
        
        # Orientacja z akcelerometru
        accel_orientation = self.accel_to_orientation(accel)
        
        # Complementary filter
        self.orientation[0] = self.alpha * gyro_orientation[0] + \
                              (1 - self.alpha) * accel_orientation[0]
        self.orientation[1] = self.alpha * gyro_orientation[1] + \
                              (1 - self.alpha) * accel_orientation[1]
        
        # Yaw z magnetometru (jeśli dostępny)
        if mag is not None:
            yaw = self.mag_to_yaw(mag, self.orientation[0], self.orientation[1])
            self.orientation[2] = yaw
        else:
            # Tylko integracja żyroskopu dla yaw
            self.orientation[2] = gyro_orientation[2]
        
        return self.orientation
    
    def accel_to_orientation(self, accel):
        """
        Oblicz roll i pitch z akcelerometru
        """
        ax, ay, az = accel
        
        roll = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        
        return np.array([roll, pitch, 0])
    
    def mag_to_yaw(self, mag, roll, pitch):
        """
        Oblicz yaw z magnetometru (skompensowany o roll/pitch)
        """
        mx, my, mz = mag
        
        # Compensate for tilt
        mag_x = mx * np.cos(pitch) + mz * np.sin(pitch)
        mag_y = mx * np.sin(roll) * np.sin(pitch) + \
                my * np.cos(roll) - \
                mz * np.sin(roll) * np.cos(pitch)
        
        yaw = np.arctan2(-mag_y, mag_x)
        
        return yaw
```

### Madgwick Filter

```python
class MadgwickFilter:
    def __init__(self, beta=0.1, sample_freq=100):
        self.beta = beta  # Convergence rate
        self.sample_freq = sample_freq
        
        # Quaternion (w, x, y, z)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
    
    def update(self, accel, gyro, mag=None):
        """
        Madgwick AHRS algorithm
        """
        dt = 1.0 / self.sample_freq
        
        # Normalize accelerometer
        accel = accel / np.linalg.norm(accel)
        
        if mag is not None:
            # Normalize magnetometer
            mag = mag / np.linalg.norm(mag)
            
            # MARG update
            self.update_marg(accel, gyro, mag, dt)
        else:
            # IMU update (no magnetometer)
            self.update_imu(accel, gyro, dt)
        
        return self.quaternion_to_euler(self.q)
    
    def update_imu(self, accel, gyro, dt):
        """
        IMU update (6 DOF)
        """
        q = self.q
        ax, ay, az = accel
        gx, gy, gz = gyro
        
        # Rate of change of quaternion from gyroscope
        qDot = 0.5 * self.quaternion_multiply(q, [0, gx, gy, gz])
        
        # Gradient descent
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - ax,
            2*(q[0]*q[1] + q[2]*q[3]) - ay,
            2*(0.5 - q[1]**2 - q[2]**2) - az
        ])
        
        J = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])
        
        step = J.T @ f
        step = step / np.linalg.norm(step)
        
        # Combine
        qDot = qDot - self.beta * step
        
        # Integrate
        self.q = q + qDot * dt
        self.q = self.q / np.linalg.norm(self.q)
    
    def quaternion_to_euler(self, q):
        """
        Konwersja quaternion -> Euler angles
        """
        w, x, y, z = q
        
        # Roll
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        
        # Pitch
        pitch = np.arcsin(2*(w*y - z*x))
        
        # Yaw
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        
        return np.array([roll, pitch, yaw])
```

## ROS2 Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3

class IMUNode(Node):
    def __init__(self):
        super().__init__('imu_node')
        
        # IMU sensor
        self.imu = IMU(sample_rate=100)
        self.filter = MadgwickFilter(beta=0.1, sample_freq=100)
        
        # Publisher
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        
        # Timer (100 Hz)
        self.timer = self.create_timer(0.01, self.publish_imu)
    
    def publish_imu(self):
        # Read IMU
        data = self.imu.read(self.get_true_state())
        
        # Filter
        orientation = self.filter.update(
            data['acceleration'],
            data['angular_velocity'],
            data['magnetic_field']
        )
        
        # Create message
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        
        # Orientation (quaternion)
        q = self.euler_to_quaternion(orientation)
        msg.orientation.w = q[0]
        msg.orientation.x = q[1]
        msg.orientation.y = q[2]
        msg.orientation.z = q[3]
        
        # Angular velocity
        msg.angular_velocity.x = data['angular_velocity'][0]
        msg.angular_velocity.y = data['angular_velocity'][1]
        msg.angular_velocity.z = data['angular_velocity'][2]
        
        # Linear acceleration
        msg.linear_acceleration.x = data['acceleration'][0]
        msg.linear_acceleration.y = data['acceleration'][1]
        msg.linear_acceleration.z = data['acceleration'][2]
        
        # Publish
        self.imu_pub.publish(msg)
```

## Popularne Chipy IMU

| Model | Akcelerometr | Żyroskop | Magnetometr | Cena |
|-------|-------------|----------|-------------|------|
| **MPU6050** | ±2-16g | ±250-2000°/s | ❌ | $2 |
| **MPU9250** | ±2-16g | ±250-2000°/s | ✅ | $5 |
| **BMI088** | ±3-24g | ±125-2000°/s | ❌ | $10 |
| **ICM-20948** | ±2-16g | ±250-2000°/s | ✅ | $8 |
| **BNO055** | ±2-16g | ±125-2000°/s | ✅ | $15 |

## Powiązane Artykuły

- [SLAM](#wiki-slam)
- [Sensor Fusion](#wiki-sensor-fusion)
- [Unitree G1](#wiki-unitree-g1)

---

*Ostatnia aktualizacja: 2025-02-11*  
*Autor: Zespół Robotyki, Laboratorium Robotów Humanoidalnych*
