# MoveIt2 - Motion Planning Framework

## Wprowadzenie

**MoveIt2** to najbardziej popularny framework do planowania ruchu w ROS2. Zapewnia narzędzia do kinematyki, planowania trajektorii, wykrywania kolizji i kontroli manipulatorów w robotyce.

## Podstawowa Konfiguracja

### Setup Assistant

```bash
# Install MoveIt2
sudo apt install ros-humble-moveit

# Run MoveIt Setup Assistant
ros2 launch moveit_setup_assistant setup_assistant.launch.py

# Kroki konfiguracji:
# 1. Load URDF
# 2. Define Self-Collision Matrix
# 3. Add Virtual Joints
# 4. Add Planning Groups
# 5. Define Robot Poses
# 6. Add End Effectors
# 7. Configure Controllers
# 8. Generate Config Files
```

### URDF Configuration

```xml
<?xml version="1.0"?>
<robot name="humanoid_arm" xmlns:xacro="http://www.ros.org/wiki/xacro">
  
  <!-- Links -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
  </link>
  
  <link name="shoulder_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Joints -->
  <joint name="shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
  </joint>
  
</robot>
```

## Python API

### Basic Motion Planning

```python
import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState

class MoveIt2Controller(Node):
    def __init__(self):
        super().__init__('moveit2_controller')
        
        # Initialize MoveItPy
        self.moveit = MoveItPy(node_name="moveit_py")
        
        # Get planning component
        self.arm = self.moveit.get_planning_component("arm")
        self.gripper = self.moveit.get_planning_component("gripper")
        
        self.get_logger().info('MoveIt2 Controller initialized')
    
    def move_to_pose(self, target_pose):
        """
        Plan and execute motion to target pose
        
        Args:
            target_pose: geometry_msgs.Pose
        """
        # Set pose target
        self.arm.set_goal_state(pose_stamped_msg=target_pose, pose_link="end_effector")
        
        # Plan
        plan_result = self.arm.plan()
        
        if plan_result:
            # Execute
            self.get_logger().info('Executing motion...')
            self.arm.execute()
        else:
            self.get_logger().error('Planning failed!')
    
    def move_to_joint_values(self, joint_values):
        """
        Move to specific joint configuration
        """
        # Set joint target
        self.arm.set_goal_state(configuration_name="home")
        # or
        self.arm.set_goal_state(joint_state=joint_values)
        
        # Plan and execute
        plan_result = self.arm.plan()
        if plan_result:
            self.arm.execute()
    
    def pick_and_place(self, pick_pose, place_pose):
        """
        Pick and place operation
        """
        # Open gripper
        self.gripper.set_goal_state(configuration_name="open")
        self.gripper.plan()
        self.gripper.execute()
        
        # Move to pre-grasp
        pre_grasp = pick_pose.copy()
        pre_grasp.position.z += 0.1
        self.move_to_pose(pre_grasp)
        
        # Move to grasp
        self.move_to_pose(pick_pose)
        
        # Close gripper
        self.gripper.set_goal_state(configuration_name="closed")
        self.gripper.plan()
        self.gripper.execute()
        
        # Lift
        self.move_to_pose(pre_grasp)
        
        # Move to place
        pre_place = place_pose.copy()
        pre_place.position.z += 0.1
        self.move_to_pose(pre_place)
        
        # Place
        self.move_to_pose(place_pose)
        
        # Open gripper
        self.gripper.set_goal_state(configuration_name="open")
        self.gripper.plan()
        self.gripper.execute()

def main():
    rclpy.init()
    node = MoveIt2Controller()
    
    # Example: move to home position
    node.move_to_joint_values([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    rclpy.spin(node)
    rclpy.shutdown()
```

## Planning Pipelines

### OMPL Configuration

```yaml
# ompl_planning.yaml
planning_plugin: ompl_interface/OMPLPlanner

planner_configs:
  RRTConnect:
    type: geometric::RRTConnect
    range: 0.0  # Max step size (0 = default)
  
  RRT:
    type: geometric::RRT
    range: 0.0
    goal_bias: 0.05
  
  PRM:
    type: geometric::PRM
    max_nearest_neighbors: 10
  
  TRRT:
    type: geometric::TRRT
    range: 0.0
    goal_bias: 0.05
    frountier_threshold: 0.0

arm:
  planner_configs:
    - RRTConnect
    - RRT
    - PRM
  projection_evaluator: joints(shoulder_pan,shoulder_lift,elbow)
  longest_valid_segment_fraction: 0.005
```

## Collision Checking

### Adding Collision Objects

```python
from moveit.planning import PlanningSceneMonitor
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive

class CollisionSceneManager:
    def __init__(self, moveit_py):
        self.moveit = moveit_py
        self.scene = self.moveit.get_planning_scene_monitor()
    
    def add_box(self, name, pose, size):
        """
        Add box collision object
        
        Args:
            name: object id
            pose: geometry_msgs.Pose
            size: [x, y, z] dimensions
        """
        # Create collision object message
        collision_object = CollisionObject()
        collision_object.header.frame_id = "world"
        collision_object.id = name
        
        # Define box
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = size
        
        collision_object.primitives.append(box)
        collision_object.primitive_poses.append(pose)
        collision_object.operation = CollisionObject.ADD
        
        # Add to scene
        self.scene.apply_collision_object(collision_object)
    
    def attach_object(self, object_name, link_name):
        """
        Attach object to robot link (e.g., gripper)
        """
        self.scene.attach_object(
            name=object_name,
            link_name=link_name,
            touch_links=["gripper_left", "gripper_right"]
        )
    
    def detach_object(self, object_name):
        """
        Detach object from robot
        """
        self.scene.detach_object(name=object_name)
```

## Cartesian Path Planning

```python
def plan_cartesian_path(self, waypoints, eef_step=0.01, jump_threshold=0.0):
    """
    Plan cartesian path through waypoints
    
    Args:
        waypoints: list of geometry_msgs.Pose
        eef_step: max distance between points (m)
        jump_threshold: max joint space jump
    """
    # Compute cartesian path
    (plan, fraction) = self.arm.compute_cartesian_path(
        waypoints=waypoints,
        eef_step=eef_step,
        jump_threshold=jump_threshold
    )
    
    if fraction < 1.0:
        self.get_logger().warn(f'Only {fraction*100}% of path achieved')
    
    return plan, fraction

def circular_motion(self, center, radius, n_points=20):
    """
    Generate circular motion
    """
    import numpy as np
    
    waypoints = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        
        pose = Pose()
        pose.position.x = center[0] + radius * np.cos(angle)
        pose.position.y = center[1] + radius * np.sin(angle)
        pose.position.z = center[2]
        
        # Keep orientation constant
        pose.orientation.w = 1.0
        
        waypoints.append(pose)
    
    return self.plan_cartesian_path(waypoints)
```

## Trajectory Execution

### Execution Monitoring

```python
class TrajectoryExecutor:
    def __init__(self, moveit_py):
        self.moveit = moveit_py
        
        # Execution monitoring
        self.execution_succeeded = False
    
    def execute_with_monitoring(self, plan):
        """
        Execute trajectory with monitoring
        """
        # Set execution callback
        def execution_callback(result):
            if result == MoveItErrorCode.SUCCESS:
                self.get_logger().info('Execution succeeded')
                self.execution_succeeded = True
            else:
                self.get_logger().error(f'Execution failed: {result}')
                self.execution_succeeded = False
        
        # Execute
        self.moveit.execute(
            plan,
            callback=execution_callback
        )
        
        return self.execution_succeeded
    
    def execute_with_replanning(self, goal, max_attempts=3):
        """
        Execute with automatic replanning on failure
        """
        for attempt in range(max_attempts):
            # Plan
            plan = self.arm.plan()
            
            if not plan:
                self.get_logger().warn(f'Planning failed (attempt {attempt+1})')
                continue
            
            # Execute
            success = self.execute_with_monitoring(plan)
            
            if success:
                return True
        
        return False
```

## Integration z ROS2

### Launch File

```python
# moveit_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("humanoid_robot")
        .robot_description(file_path="config/humanoid.urdf.xacro")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    
    # Move Group Node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )
    
    # RViz
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", moveit_config.package_path / "config/moveit.rviz"],
    )
    
    return LaunchDescription([move_group_node, rviz_node])
```

## Powiązane Artykuły

- [ROS2](#wiki-ros2)
- [Motion Planning](#wiki-motion-planning)
- [Manipulation](#wiki-manipulation)
- [Kinematics](#wiki-kinematics)

---

*Ostatnia aktualizacja: 2025-02-12*  
*Autor: Zespół Interakcji, Laboratorium Robotów Humanoidalnych*
