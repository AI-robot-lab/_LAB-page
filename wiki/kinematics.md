# Kinematyka Robotów

## Wprowadzenie

**Kinematyka** opisuje ruch robota bez uwzględniania sił. Obejmuje kinematykę prostą (FK) i odwrotną (IK), kluczowe dla kontroli robotów manipulacyjnych i humanoidalnych.

## Forward Kinematics

### DH Parameters

```python
import numpy as np

class DHParameters:
    def __init__(self, a, alpha, d, theta):
        """
        Denavit-Hartenberg parameters
        
        a: link length
        alpha: link twist
        d: link offset
        theta: joint angle
        """
        self.a = a
        self.alpha = alpha
        self.d = d
        self.theta = theta
    
    def transformation_matrix(self):
        """
        Compute 4x4 transformation matrix
        """
        ct = np.cos(self.theta)
        st = np.sin(self.theta)
        ca = np.cos(self.alpha)
        sa = np.sin(self.alpha)
        
        T = np.array([
            [ct, -st*ca,  st*sa, self.a*ct],
            [st,  ct*ca, -ct*sa, self.a*st],
            [0,   sa,     ca,     self.d],
            [0,   0,      0,      1]
        ])
        
        return T

class RobotArm:
    def __init__(self, dh_params):
        """
        dh_params: list of DHParameters
        """
        self.dh_params = dh_params
        self.n_joints = len(dh_params)
    
    def forward_kinematics(self, joint_angles):
        """
        Compute end-effector pose from joint angles
        
        Returns: 4x4 transformation matrix
        """
        T = np.eye(4)
        
        for i, angle in enumerate(joint_angles):
            # Update theta
            self.dh_params[i].theta = angle
            
            # Multiply transformations
            T_i = self.dh_params[i].transformation_matrix()
            T = T @ T_i
        
        return T
    
    def get_position(self, T):
        """Extract position from transformation matrix"""
        return T[:3, 3]
    
    def get_orientation(self, T):
        """Extract rotation matrix"""
        return T[:3, :3]

# Example: 3-DOF planar arm
dh_params = [
    DHParameters(a=1.0, alpha=0, d=0, theta=0),
    DHParameters(a=1.0, alpha=0, d=0, theta=0),
    DHParameters(a=0.5, alpha=0, d=0, theta=0)
]

arm = RobotArm(dh_params)
T = arm.forward_kinematics([np.pi/4, np.pi/3, -np.pi/6])
print("End-effector position:", arm.get_position(T))
```

## Inverse Kinematics

### Analytical IK (2-link planar)

```python
class PlanarArmIK:
    def __init__(self, l1, l2):
        self.l1 = l1  # Link 1 length
        self.l2 = l2  # Link 2 length
    
    def inverse_kinematics(self, x, y):
        """
        Analytical IK for 2-link planar arm
        
        Returns: (theta1, theta2) in radians
        """
        # Distance to target
        r = np.sqrt(x**2 + y**2)
        
        # Check reachability
        if r > (self.l1 + self.l2) or r < abs(self.l1 - self.l2):
            raise ValueError("Target unreachable")
        
        # Law of cosines
        cos_theta2 = (r**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        
        # Elbow up configuration
        theta2 = np.arccos(cos_theta2)
        
        # theta1
        alpha = np.arctan2(y, x)
        beta = np.arctan2(self.l2 * np.sin(theta2), 
                          self.l1 + self.l2 * np.cos(theta2))
        theta1 = alpha - beta
        
        return theta1, theta2

# Usage
ik = PlanarArmIK(l1=1.0, l2=1.0)
theta1, theta2 = ik.inverse_kinematics(x=1.5, y=0.5)
print(f"Joint angles: θ1={np.degrees(theta1):.1f}°, θ2={np.degrees(theta2):.1f}°")
```

### Numerical IK (Jacobian-based)

```python
class NumericalIK:
    def __init__(self, robot):
        self.robot = robot
        self.epsilon = 1e-4
    
    def inverse_kinematics(self, target_pos, q0=None, max_iter=100, tol=1e-3):
        """
        Numerical IK using Jacobian pseudo-inverse
        
        Args:
            target_pos: desired end-effector position [x, y, z]
            q0: initial joint configuration
            
        Returns: joint angles
        """
        if q0 is None:
            q0 = np.zeros(self.robot.n_joints)
        
        q = q0.copy()
        
        for i in range(max_iter):
            # Current position
            T = self.robot.forward_kinematics(q)
            current_pos = self.robot.get_position(T)
            
            # Error
            error = target_pos - current_pos
            
            if np.linalg.norm(error) < tol:
                return q  # Converged
            
            # Compute Jacobian
            J = self.compute_jacobian(q)
            
            # Pseudo-inverse
            J_pinv = np.linalg.pinv(J)
            
            # Update
            dq = J_pinv @ error
            q = q + 0.1 * dq  # Damping factor
            
            # Joint limits
            q = np.clip(q, -np.pi, np.pi)
        
        raise ValueError("IK did not converge")
    
    def compute_jacobian(self, q):
        """
        Numerical Jacobian computation
        """
        J = np.zeros((3, self.robot.n_joints))
        
        # Current position
        T0 = self.robot.forward_kinematics(q)
        pos0 = self.robot.get_position(T0)
        
        # Finite differences
        for i in range(self.robot.n_joints):
            q_plus = q.copy()
            q_plus[i] += self.epsilon
            
            T_plus = self.robot.forward_kinematics(q_plus)
            pos_plus = self.robot.get_position(T_plus)
            
            J[:, i] = (pos_plus - pos0) / self.epsilon
        
        return J
```

## Velocity Kinematics

### Geometric Jacobian

```python
class VelocityKinematics:
    def __init__(self, robot):
        self.robot = robot
    
    def geometric_jacobian(self, q):
        """
        Compute geometric Jacobian
        
        v = J * q_dot
        """
        J = np.zeros((6, self.robot.n_joints))
        
        # Get all transformation matrices
        T_list = []
        T = np.eye(4)
        
        for i in range(self.robot.n_joints):
            self.robot.dh_params[i].theta = q[i]
            T_i = self.robot.dh_params[i].transformation_matrix()
            T = T @ T_i
            T_list.append(T.copy())
        
        # End-effector position
        p_n = T[:3, 3]
        
        # Compute each column of Jacobian
        for i in range(self.robot.n_joints):
            if i == 0:
                T_i = np.eye(4)
            else:
                T_i = T_list[i-1]
            
            # Joint axis (assuming revolute joints around Z)
            z_i = T_i[:3, 2]
            p_i = T_i[:3, 3]
            
            # Linear velocity part
            J[:3, i] = np.cross(z_i, p_n - p_i)
            
            # Angular velocity part
            J[3:, i] = z_i
        
        return J
    
    def manipulability(self, q):
        """
        Yoshikawa manipulability measure
        
        μ = √det(J J^T)
        """
        J = self.geometric_jacobian(q)
        
        # Use only position part
        J_pos = J[:3, :]
        
        manipulability = np.sqrt(np.linalg.det(J_pos @ J_pos.T))
        
        return manipulability
```

## Singularities

```python
def check_singularity(J, threshold=0.01):
    """
    Check if configuration is near singularity
    
    Uses condition number of Jacobian
    """
    # Singular values
    U, s, Vt = np.linalg.svd(J)
    
    # Condition number
    cond = s[0] / s[-1]
    
    # Manipulability
    manipulability = np.prod(s)
    
    is_singular = manipulability < threshold
    
    return is_singular, cond, manipulability
```

## Powiązane Artykuły

- [Unitree G1](#wiki-unitree-g1)
- [Motion Planning](#wiki-motion-planning)
- [Control Theory](#wiki-control-theory)

---

*Ostatnia aktualizacja: 2025-02-11*
