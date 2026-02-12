# Motion Planning - Planowanie Ruchu

## Wprowadzenie

**Motion Planning** to proces wyznaczania trajektorii ruchu robota z pozycji startowej do docelowej, z unikaniem przeszkód. W robotyce humanoidalnej obejmuje planowanie chodu, manipulacji i całego ciała.

## Sampling-Based Methods

### RRT (Rapidly-exploring Random Tree)

```python
import numpy as np
import matplotlib.pyplot as plt

class RRT:
    def __init__(self, start, goal, obstacles, bounds, step_size=0.5, max_iter=5000):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        
        # Tree
        self.nodes = [self.start]
        self.parents = {tuple(self.start): None}
    
    def plan(self):
        """
        Plan path using RRT
        """
        for i in range(self.max_iter):
            # Sample random point
            if np.random.rand() < 0.1:
                # 10% chance to sample goal
                rand_point = self.goal
            else:
                rand_point = self.sample_random_point()
            
            # Find nearest node
            nearest_node = self.nearest(rand_point)
            
            # Steer towards random point
            new_node = self.steer(nearest_node, rand_point)
            
            # Check collision
            if not self.is_collision_free(nearest_node, new_node):
                continue
            
            # Add to tree
            self.nodes.append(new_node)
            self.parents[tuple(new_node)] = nearest_node
            
            # Check if goal reached
            if np.linalg.norm(new_node - self.goal) < self.step_size:
                self.parents[tuple(self.goal)] = new_node
                return self.extract_path()
        
        return None  # No path found
    
    def sample_random_point(self):
        """
        Sample random point in configuration space
        """
        return np.random.uniform(self.bounds[0], self.bounds[1])
    
    def nearest(self, point):
        """
        Find nearest node in tree
        """
        distances = [np.linalg.norm(node - point) for node in self.nodes]
        return self.nodes[np.argmin(distances)]
    
    def steer(self, from_node, to_point):
        """
        Steer from node towards point
        """
        direction = to_point - from_node
        distance = np.linalg.norm(direction)
        
        if distance < self.step_size:
            return to_point
        
        direction = direction / distance
        return from_node + direction * self.step_size
    
    def is_collision_free(self, from_node, to_node):
        """
        Check if path segment is collision-free
        """
        # Sample points along segment
        n_samples = int(np.linalg.norm(to_node - from_node) / 0.1)
        
        for i in range(n_samples + 1):
            t = i / n_samples if n_samples > 0 else 0
            point = from_node + t * (to_node - from_node)
            
            # Check against obstacles
            for obstacle in self.obstacles:
                if self.point_in_obstacle(point, obstacle):
                    return False
        
        return True
    
    def point_in_obstacle(self, point, obstacle):
        """
        Check if point is inside obstacle (circle)
        """
        center, radius = obstacle
        return np.linalg.norm(point - center) < radius
    
    def extract_path(self):
        """
        Extract path from tree
        """
        path = [self.goal]
        current = self.goal
        
        while tuple(current) in self.parents and self.parents[tuple(current)] is not None:
            current = self.parents[tuple(current)]
            path.append(current)
        
        return path[::-1]
```

### RRT*

```python
class RRTStar(RRT):
    def __init__(self, *args, search_radius=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_radius = search_radius
        self.costs = {tuple(self.start): 0.0}
    
    def plan(self):
        """
        RRT* with path optimization
        """
        for i in range(self.max_iter):
            rand_point = self.sample_random_point() if np.random.rand() > 0.1 else self.goal
            
            nearest_node = self.nearest(rand_point)
            new_node = self.steer(nearest_node, rand_point)
            
            if not self.is_collision_free(nearest_node, new_node):
                continue
            
            # Find near nodes
            near_nodes = self.find_near_nodes(new_node)
            
            # Choose best parent
            min_cost = self.costs[tuple(nearest_node)] + np.linalg.norm(new_node - nearest_node)
            best_parent = nearest_node
            
            for near_node in near_nodes:
                cost = self.costs[tuple(near_node)] + np.linalg.norm(new_node - near_node)
                
                if cost < min_cost and self.is_collision_free(near_node, new_node):
                    min_cost = cost
                    best_parent = near_node
            
            # Add new node
            self.nodes.append(new_node)
            self.parents[tuple(new_node)] = best_parent
            self.costs[tuple(new_node)] = min_cost
            
            # Rewire tree
            self.rewire(new_node, near_nodes)
            
            if np.linalg.norm(new_node - self.goal) < self.step_size:
                self.parents[tuple(self.goal)] = new_node
                self.costs[tuple(self.goal)] = min_cost + np.linalg.norm(self.goal - new_node)
                return self.extract_path()
        
        return None
    
    def find_near_nodes(self, node):
        """
        Find nodes within search radius
        """
        return [n for n in self.nodes 
                if np.linalg.norm(n - node) < self.search_radius]
    
    def rewire(self, new_node, near_nodes):
        """
        Rewire tree for optimization
        """
        for near_node in near_nodes:
            new_cost = self.costs[tuple(new_node)] + np.linalg.norm(near_node - new_node)
            
            if new_cost < self.costs.get(tuple(near_node), float('inf')):
                if self.is_collision_free(new_node, near_node):
                    self.parents[tuple(near_node)] = new_node
                    self.costs[tuple(near_node)] = new_cost
```

## Graph-Based Methods

### A* Algorithm

```python
import heapq

class AStar:
    def __init__(self, grid, start, goal):
        self.grid = grid  # 2D occupancy grid
        self.start = start
        self.goal = goal
        
        self.height, self.width = grid.shape
    
    def plan(self):
        """
        A* path planning
        """
        # Priority queue: (f_score, node)
        open_set = []
        heapq.heappush(open_set, (0, self.start))
        
        # Tracking
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.heuristic(self.start, self.goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == self.goal:
                return self.reconstruct_path(came_from, current)
            
            # Check neighbors
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + self.cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, self.goal)
                    
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def heuristic(self, a, b):
        """
        Euclidean distance heuristic
        """
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def cost(self, a, b):
        """
        Cost to move from a to b
        """
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, node):
        """
        Get valid neighbors (8-connected)
        """
        x, y = node
        neighbors = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    # Check obstacle
                    if self.grid[nx, ny] == 0:
                        neighbors.append((nx, ny))
        
        return neighbors
    
    def reconstruct_path(self, came_from, current):
        """
        Reconstruct path from came_from dict
        """
        path = [current]
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        return path[::-1]
```

## Trajectory Optimization

### Minimum Jerk Trajectory

```python
class MinimumJerkTrajectory:
    def __init__(self, start_pos, end_pos, duration):
        self.start = start_pos
        self.end = end_pos
        self.duration = duration
    
    def compute(self, t):
        """
        Compute position at time t using 5th order polynomial
        
        Boundary conditions:
        - position, velocity, acceleration at start and end
        """
        if t >= self.duration:
            return self.end
        
        # Normalized time
        tau = t / self.duration
        
        # 5th order polynomial
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        
        # Interpolate
        position = self.start + (self.end - self.start) * s
        
        return position
    
    def velocity(self, t):
        """
        Compute velocity
        """
        if t >= self.duration:
            return 0.0
        
        tau = t / self.duration
        
        # Derivative of s
        ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.duration
        
        velocity = (self.end - self.start) * ds
        
        return velocity
    
    def acceleration(self, t):
        """
        Compute acceleration
        """
        if t >= self.duration:
            return 0.0
        
        tau = t / self.duration
        
        # Second derivative
        dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / (self.duration**2)
        
        acceleration = (self.end - self.start) * dds
        
        return acceleration
```

## Whole-Body Planning

### Inverse Kinematics + Motion Planning

```python
class WholeBodyPlanner:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.ik_solver = IKSolver(robot_model)
    
    def plan_reaching(self, hand_target, obstacles=[]):
        """
        Plan reaching motion for humanoid arm
        """
        # Sample configurations
        configs = []
        
        for i in range(1000):
            # Random joint angles
            config = self.sample_config()
            
            # Check IK feasibility
            hand_pos = self.robot.forward_kinematics(config)
            
            if np.linalg.norm(hand_pos - hand_target) < 0.1:
                # Check collision
                if not self.is_in_collision(config, obstacles):
                    configs.append(config)
        
        if not configs:
            return None
        
        # Use RRT to connect start to one of valid configs
        rrt = RRT(
            start=self.robot.current_config,
            goal=configs[0],
            obstacles=obstacles,
            bounds=self.robot.joint_limits
        )
        
        return rrt.plan()
    
    def sample_config(self):
        """
        Sample random joint configuration
        """
        return np.random.uniform(
            self.robot.joint_limits[:, 0],
            self.robot.joint_limits[:, 1]
        )
```

## ROS2 Integration

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class MotionPlannerNode(Node):
    def __init__(self):
        super().__init__('motion_planner')
        
        # Planner
        self.planner = RRTStar(
            start=[0, 0],
            goal=[10, 10],
            obstacles=[],
            bounds=[[0, 0], [10, 10]]
        )
        
        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        
        # Publishers
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
    
    def goal_callback(self, msg):
        """
        Plan path to goal
        """
        goal = [msg.pose.position.x, msg.pose.position.y]
        
        self.planner.goal = np.array(goal)
        
        # Plan
        path = self.planner.plan()
        
        if path:
            self.publish_path(path)
    
    def publish_path(self, path):
        """
        Publish path as ROS message
        """
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        for point in path:
            pose = PoseStamped()
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            msg.poses.append(pose)
        
        self.path_pub.publish(msg)
```

## Porównanie Algorytmów

| Algorytm | Kompletność | Optymalność | Złożoność | Use Case |
|----------|-------------|-------------|-----------|----------|
| **RRT** | Probabilistic | ❌ | O(n log n) | Fast planning |
| **RRT*** | Probabilistic | Asymptotic | O(n log n) | Optimal paths |
| **A*** | ✅ | ✅ | O(b^d) | Grid-based |
| **Dijkstra** | ✅ | ✅ | O(V²) | Known graph |
| **PRM** | Probabilistic | ❌ | O(n log n) | Multi-query |

## Powiązane Artykuły

- [Kinematics](#wiki-kinematics)
- [Control Theory](#wiki-control-theory)
- [Trajectory Optimization](#wiki-trajectory-optimization)
- [SLAM](#wiki-slam)

---

*Ostatnia aktualizacja: 2025-02-11*  
*Autor: Zespół Akcji, Laboratorium Robotów Humanoidalnych*
