# Optymalizacja Trajektorii

## Wprowadzenie

**Optymalizacja trajektorii** to proces znajdowania optymalnej ścieżki ruchu robota spełniającej określone kryteria (czas, energia, gładkość) przy zachowaniu ograniczeń fizycznych i bezpieczeństwa.

## Minimum Jerk Trajectory

### 5th Order Polynomial

```python
import numpy as np

class MinimumJerkTrajectory:
    def __init__(self, start, goal, duration):
        """
        Minimum jerk trajectory
        
        Args:
            start: [pos, vel, acc] w t=0
            goal: [pos, vel, acc] w t=T
            duration: czas trwania T
        """
        self.start = start
        self.goal = goal
        self.T = duration
        
        # Oblicz współczynniki wielomianu 5. stopnia
        self.coefficients = self.compute_coefficients()
    
    def compute_coefficients(self):
        """
        Polynomial: s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        """
        p0, v0, a0 = self.start
        pT, vT, aT = self.goal
        T = self.T
        
        # Macierz układu równań
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        ])
        
        # Wektor wymagań
        b = np.array([p0, v0, a0, pT, vT, aT])
        
        # Rozwiąż
        coeffs = np.linalg.solve(A, b)
        
        return coeffs
    
    def position(self, t):
        """Pozycja w czasie t"""
        c = self.coefficients
        return (c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + 
                c[4]*t**4 + c[5]*t**5)
    
    def velocity(self, t):
        """Prędkość w czasie t"""
        c = self.coefficients
        return (c[1] + 2*c[2]*t + 3*c[3]*t**2 + 
                4*c[4]*t**3 + 5*c[5]*t**4)
    
    def acceleration(self, t):
        """Przyspieszenie w czasie t"""
        c = self.coefficients
        return (2*c[2] + 6*c[3]*t + 12*c[4]*t**2 + 20*c[5]*t**3)
    
    def jerk(self, t):
        """Jerk (pochodna przyspieszenia) w czasie t"""
        c = self.coefficients
        return (6*c[3] + 24*c[4]*t + 60*c[5]*t**2)

# Przykład użycia
start = [0.0, 0.0, 0.0]  # pos, vel, acc
goal = [1.0, 0.0, 0.0]
traj = MinimumJerkTrajectory(start, goal, duration=2.0)

# Sample trajectory
t_samples = np.linspace(0, 2.0, 100)
positions = [traj.position(t) for t in t_samples]
velocities = [traj.velocity(t) for t in t_samples]
```

## Direct Collocation

### Trajectory Optimization Framework

```python
from scipy.optimize import minimize
import casadi as ca

class DirectCollocation:
    def __init__(self, dynamics, cost_function, constraints):
        self.dynamics = dynamics
        self.cost = cost_function
        self.constraints = constraints
    
    def optimize(self, x0, xf, N=50, T=5.0):
        """
        Optymalizacja trajektorii metodą direct collocation
        
        Args:
            x0: stan początkowy
            xf: stan końcowy
            N: liczba punktów kolokacji
            T: czas trwania
        """
        dt = T / N
        
        # Decision variables: [x0, u0, x1, u1, ..., xN, uN]
        n_states = len(x0)
        n_controls = self.dynamics.n_controls
        
        n_vars = (N + 1) * n_states + N * n_controls
        
        # Funkcja celu
        def objective(z):
            cost = 0
            for k in range(N):
                x_k = z[k*(n_states+n_controls):k*(n_states+n_controls)+n_states]
                u_k = z[k*(n_states+n_controls)+n_states:(k+1)*(n_states+n_controls)]
                
                # Running cost
                cost += self.cost(x_k, u_k) * dt
            
            # Terminal cost
            x_N = z[-n_states:]
            cost += self.cost(x_N, np.zeros(n_controls))
            
            return cost
        
        # Ograniczenia
        def collocation_constraints(z):
            constraints = []
            
            for k in range(N):
                x_k = z[k*(n_states+n_controls):k*(n_states+n_controls)+n_states]
                u_k = z[k*(n_states+n_controls)+n_states:(k+1)*(n_states+n_controls)]
                x_next = z[(k+1)*(n_states+n_controls):(k+1)*(n_states+n_controls)+n_states]
                
                # Dynamics constraint: x_{k+1} = x_k + f(x_k, u_k) * dt
                f_k = self.dynamics(x_k, u_k)
                constraint = x_next - (x_k + f_k * dt)
                constraints.extend(constraint)
            
            return np.array(constraints)
        
        # Initial guess
        z0 = np.zeros(n_vars)
        
        # Boundary conditions
        bounds = []
        for k in range(N + 1):
            if k == 0:
                # Fix initial state
                for i in range(n_states):
                    bounds.append((x0[i], x0[i]))
            elif k == N:
                # Fix final state
                for i in range(n_states):
                    bounds.append((xf[i], xf[i]))
            else:
                # Free states
                for i in range(n_states):
                    bounds.append((None, None))
            
            # Control bounds
            if k < N:
                for i in range(n_controls):
                    bounds.append((-10, 10))  # Control limits
        
        # Optimize
        result = minimize(
            objective,
            z0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': collocation_constraints}
        )
        
        return result.x
```

## Differential Dynamic Programming (DDP)

```python
class DDP:
    def __init__(self, dynamics, cost, horizon=50):
        self.f = dynamics  # x_next = f(x, u)
        self.l = cost      # stage cost
        self.N = horizon
    
    def backward_pass(self, X, U):
        """
        Backward pass - oblicz gains
        """
        # Terminal cost derivatives
        V_x = self.l_x(X[-1], None)  # ∂l/∂x
        V_xx = self.l_xx(X[-1], None)  # ∂²l/∂x²
        
        K = []  # Feedback gains
        k = []  # Feedforward terms
        
        for t in reversed(range(self.N)):
            x_t = X[t]
            u_t = U[t]
            
            # Dynamics derivatives
            f_x = self.f_x(x_t, u_t)  # ∂f/∂x
            f_u = self.f_u(x_t, u_t)  # ∂f/∂u
            
            # Cost derivatives
            l_x = self.l_x(x_t, u_t)
            l_u = self.l_u(x_t, u_t)
            l_xx = self.l_xx(x_t, u_t)
            l_uu = self.l_uu(x_t, u_t)
            l_ux = self.l_ux(x_t, u_t)
            
            # Q-function derivatives
            Q_x = l_x + f_x.T @ V_x
            Q_u = l_u + f_u.T @ V_x
            Q_xx = l_xx + f_x.T @ V_xx @ f_x
            Q_uu = l_uu + f_u.T @ V_xx @ f_u
            Q_ux = l_ux + f_u.T @ V_xx @ f_x
            
            # Gains
            K_t = -np.linalg.inv(Q_uu) @ Q_ux
            k_t = -np.linalg.inv(Q_uu) @ Q_u
            
            K.append(K_t)
            k.append(k_t)
            
            # Update value function
            V_x = Q_x + K_t.T @ Q_uu @ k_t + K_t.T @ Q_u + Q_ux.T @ k_t
            V_xx = Q_xx + K_t.T @ Q_uu @ K_t + K_t.T @ Q_ux + Q_ux.T @ K_t
        
        return list(reversed(K)), list(reversed(k))
    
    def forward_pass(self, X, U, K, k, alpha=1.0):
        """
        Forward pass - symuluj z nowymi kontrolami
        """
        X_new = [X[0]]
        U_new = []
        
        for t in range(self.N):
            # New control with line search
            u_new = U[t] + alpha * k[t] + K[t] @ (X_new[t] - X[t])
            U_new.append(u_new)
            
            # Simulate
            x_next = self.f(X_new[t], u_new)
            X_new.append(x_next)
        
        return X_new, U_new
```

## CHOMP (Covariant Hamiltonian Optimization)

```python
class CHOMP:
    def __init__(self, start, goal, obstacles, n_waypoints=20):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.n = n_waypoints
        
        # Initialize trajectory
        self.theta = self.initialize_trajectory()
    
    def initialize_trajectory(self):
        """
        Linearna interpolacja między start a goal
        """
        theta = np.linspace(self.start, self.goal, self.n)
        return theta
    
    def optimize(self, max_iter=100):
        """
        CHOMP optimization
        """
        for iteration in range(max_iter):
            # Compute gradient
            grad = self.compute_gradient()
            
            # Update trajectory
            self.theta -= 0.01 * grad
            
            # Check convergence
            if np.linalg.norm(grad) < 1e-3:
                break
        
        return self.theta
    
    def compute_gradient(self):
        """
        Gradient = smoothness cost + obstacle cost
        """
        # Smoothness gradient (finite differences)
        grad_smooth = self.smoothness_gradient()
        
        # Obstacle gradient
        grad_obs = self.obstacle_gradient()
        
        # Combined
        gradient = grad_smooth + 10.0 * grad_obs
        
        return gradient
    
    def smoothness_gradient(self):
        """
        Gradient kosztu gładkości
        """
        grad = np.zeros_like(self.theta)
        
        for i in range(1, self.n - 1):
            # Finite difference approximation
            grad[i] = 2*self.theta[i] - self.theta[i-1] - self.theta[i+1]
        
        return grad
    
    def obstacle_gradient(self):
        """
        Gradient kosztu kolizji
        """
        grad = np.zeros_like(self.theta)
        
        for i in range(self.n):
            point = self.theta[i]
            
            # Dla każdej przeszkody
            for obstacle in self.obstacles:
                distance = self.distance_to_obstacle(point, obstacle)
                
                if distance < obstacle['radius']:
                    # Gradient odpychający
                    direction = (point - obstacle['center']) / distance
                    grad[i] += direction * (1/distance - 1/obstacle['radius'])
        
        return grad
```

## Powiązane Artykuły

- [Motion Planning](#wiki-motion-planning)
- [Control Theory](#wiki-control-theory)
- [Kinematics](#wiki-kinematics)

---

*Ostatnia aktualizacja: 2025-02-12*  
*Autor: Zespół Interakcji, Laboratorium Robotów Humanoidalnych*
