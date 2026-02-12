# Control Theory - Teoria Sterowania

## Wprowadzenie

**Control Theory** dostarcza narzędzi do sterowania systemami dynamicznymi. W robotyce humanoidalnej wykorzystywana do kontroli pozycji, prędkości i siły.

## PID Controller

```python
class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits=None):
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        
        self.output_limits = output_limits
        
        # State
        self.integral = 0.0
        self.last_error = 0.0
    
    def update(self, setpoint, measurement, dt):
        """
        Compute control output
        
        u(t) = Kp*e(t) + Ki*∫e(τ)dτ + Kd*de(t)/dt
        """
        # Error
        error = setpoint - measurement
        
        # Proportional
        P = self.Kp * error
        
        # Integral
        self.integral += error * dt
        I = self.Ki * self.integral
        
        # Derivative
        D = self.Kd * (error - self.last_error) / dt if dt > 0 else 0.0
        
        # Control output
        output = P + I + D
        
        # Apply limits
        if self.output_limits:
            output = np.clip(output, *self.output_limits)
            
            # Anti-windup
            if abs(output) >= abs(self.output_limits[1]):
                self.integral -= error * dt
        
        # Update state
        self.last_error = error
        
        return output
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.last_error = 0.0

# Example: Joint position control
joint_controller = PIDController(
    Kp=100.0,
    Ki=10.0,
    Kd=5.0,
    output_limits=(-50, 50)  # Torque limits
)

target_angle = np.pi / 4
current_angle = 0.0
dt = 0.01

for _ in range(1000):
    torque = joint_controller.update(target_angle, current_angle, dt)
    
    # Simulate dynamics (simplified)
    current_angle += torque * dt * 0.01
```

## LQR (Linear Quadratic Regulator)

```python
import scipy.linalg

class LQRController:
    def __init__(self, A, B, Q, R):
        """
        LQR for linear system: ẋ = Ax + Bu
        
        Cost: J = ∫(x'Qx + u'Ru)dt
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        
        # Solve Riccati equation
        self.P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        
        # Optimal gain: K = R^(-1)B'P
        self.K = np.linalg.inv(R) @ B.T @ self.P
    
    def control(self, x, x_ref=None):
        """
        Compute optimal control: u = -K(x - x_ref)
        """
        if x_ref is None:
            x_ref = np.zeros_like(x)
        
        u = -self.K @ (x - x_ref)
        
        return u

# Example: Cart-pole
# State: [position, velocity, angle, angular_velocity]
# Control: horizontal force

m_cart = 1.0  # kg
m_pole = 0.1
l = 0.5  # m
g = 9.81

# Linearized dynamics around upright position
A = np.array([
    [0, 1, 0, 0],
    [0, 0, -m_pole*g/m_cart, 0],
    [0, 0, 0, 1],
    [0, 0, (m_cart+m_pole)*g/(l*m_cart), 0]
])

B = np.array([
    [0],
    [1/m_cart],
    [0],
    [-1/(l*m_cart)]
])

# Cost matrices
Q = np.diag([10.0, 1.0, 100.0, 1.0])  # State cost
R = np.array([[0.1]])  # Control cost

lqr = LQRController(A, B, Q, R)

# Control
state = np.array([0.1, 0.0, 0.05, 0.0])
force = lqr.control(state)
```

## MPC (Model Predictive Control)

```python
from scipy.optimize import minimize

class MPCController:
    def __init__(self, A, B, Q, R, N=10):
        """
        Model Predictive Control
        
        Args:
            N: prediction horizon
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N = N
        
        self.n = A.shape[0]  # state dim
        self.m = B.shape[1]  # control dim
    
    def predict_trajectory(self, x0, U):
        """
        Predict state trajectory given control sequence
        """
        X = [x0]
        x = x0
        
        for u in U:
            x = self.A @ x + self.B @ u
            X.append(x)
        
        return np.array(X)
    
    def cost_function(self, U, x0, x_ref):
        """
        Cost function to minimize
        """
        # Reshape control sequence
        U = U.reshape(self.N, self.m)
        
        # Predict trajectory
        X = self.predict_trajectory(x0, U)
        
        # Compute cost
        cost = 0.0
        
        for i in range(self.N):
            x_error = X[i] - x_ref
            cost += x_error.T @ self.Q @ x_error
            cost += U[i].T @ self.R @ U[i]
        
        # Terminal cost
        x_error = X[-1] - x_ref
        cost += x_error.T @ self.Q @ x_error
        
        return cost
    
    def control(self, x0, x_ref):
        """
        Compute optimal control sequence
        """
        # Initial guess
        U0 = np.zeros(self.N * self.m)
        
        # Optimize
        result = minimize(
            self.cost_function,
            U0,
            args=(x0, x_ref),
            method='SLSQP'
        )
        
        # Extract first control
        U_opt = result.x.reshape(self.N, self.m)
        
        return U_opt[0]
```

## Impedance Control

```python
class ImpedanceController:
    def __init__(self, M_d, D_d, K_d):
        """
        Impedance control for compliant manipulation
        
        M_d: desired inertia
        D_d: desired damping
        K_d: desired stiffness
        """
        self.M_d = M_d
        self.D_d = D_d
        self.K_d = K_d
    
    def control(self, x, x_d, v, v_d, f_ext):
        """
        Compute control force
        
        M_d*(ẍ_d - ẍ) + D_d*(ẋ_d - ẋ) + K_d*(x_d - x) = f_ext
        """
        # Position error
        e_pos = x_d - x
        
        # Velocity error
        e_vel = v_d - v
        
        # Desired acceleration (simplified, no ẍ_d term)
        a_d = 0
        
        # Control force
        f_control = self.M_d * a_d + self.D_d * e_vel + self.K_d * e_pos - f_ext
        
        return f_control
```

## Adaptive Control

```python
class AdaptiveController:
    def __init__(self, gamma=0.1):
        self.gamma = gamma  # Adaptation rate
        self.theta_hat = np.zeros(3)  # Parameter estimates
    
    def update(self, x, u, x_dot, dt):
        """
        Parameter adaptation using gradient descent
        """
        # Regressor (system-specific)
        Y = np.array([x, u, 1])
        
        # Prediction error
        x_dot_pred = Y @ self.theta_hat
        error = x_dot - x_dot_pred
        
        # Update parameters
        self.theta_hat += self.gamma * Y * error * dt
        
        return self.theta_hat
```

## Powiązane Artykuły

- [Motion Planning](#wiki-motion-planning)
- [Kinematics](#wiki-kinematics)
- [Reinforcement Learning](#wiki-reinforcement-learning)

---

*Ostatnia aktualizacja: 2025-02-11*
