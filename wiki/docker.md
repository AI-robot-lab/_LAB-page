# Docker dla Robotyki

## Wprowadzenie

**Docker** umożliwia konteneryzację aplikacji robotycznych, zapewniając spójność środowisk między urządzeniami i ułatwiając deployment.

## Podstawy Docker

### Dockerfile dla ROS2

```dockerfile
FROM ros:humble-ros-base

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip3 install \
    numpy \
    torch \
    ultralytics

# Create workspace
WORKDIR /ros2_ws
RUN mkdir -p src

# Copy source code
COPY ./src /ros2_ws/src

# Build workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# Source setup
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

CMD ["bash"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  robot:
    build: .
    image: humanoid_robot:latest
    container_name: robot_main
    network_mode: host
    privileged: true
    
    volumes:
      - ./src:/ros2_ws/src
      - /dev:/dev
    
    devices:
      - /dev/video0:/dev/video0  # Camera
      - /dev/ttyUSB0:/dev/ttyUSB0  # Serial
    
    environment:
      - DISPLAY=$DISPLAY
      - ROS_DOMAIN_ID=0
    
    command: ros2 launch robot_bringup robot.launch.py

  simulation:
    image: nvidia/isaac-sim:2023.1.1
    container_name: isaac_sim
    runtime: nvidia
    
    volumes:
      - ./models:/workspace/models
    
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
    
    ports:
      - "8211:8211"
```

## Build & Run

```bash
# Build image
docker build -t humanoid_robot:latest .

# Run container
docker run -it --rm \
  --name robot \
  --network host \
  --privileged \
  -v /dev:/dev \
  -e DISPLAY=$DISPLAY \
  humanoid_robot:latest

# With docker-compose
docker-compose up -d

# View logs
docker-compose logs -f robot

# Stop
docker-compose down
```

## Multi-Stage Build

```dockerfile
# Stage 1: Build
FROM ros:humble-ros-base AS builder

WORKDIR /ros2_ws
COPY ./src ./src

RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# Stage 2: Runtime
FROM ros:humble-ros-base-jammy

# Copy built workspace
COPY --from=builder /ros2_ws/install /ros2_ws/install

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws
CMD ["bash"]
```

## GPU Support

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install ROS2
RUN apt-get update && apt-get install -y curl gnupg2 lsb-release
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    python3-colcon-common-extensions

# PyTorch with CUDA
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

CMD ["bash"]
```

```bash
# Run with GPU
docker run --gpus all -it robot:latest
```

## Networking

```bash
# Host network (share network with host)
docker run --network host robot:latest

# Bridge network (create custom network)
docker network create robot_net

docker run --network robot_net --name robot1 robot:latest
docker run --network robot_net --name robot2 robot:latest

# ROS2 communication between containers
docker run -e ROS_DOMAIN_ID=0 --network host robot:latest
```

## Powiązane Artykuły

- [ROS2](#wiki-ros2)
- [Isaac Lab](#wiki-isaac-lab)

---

*Ostatnia aktualizacja: 2025-02-11*
