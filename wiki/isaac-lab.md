# NVIDIA Isaac Lab

## Wprowadzenie

**NVIDIA Isaac Lab** (dawniej Isaac Gym) to platforma do uczenia przez wzmacnianie (Reinforcement Learning) robotów w symulacji, zbudowana na silniku NVIDIA Omniverse i PhysX 5.

Jest to narzędzie klasy enterprise do:
- Trenowania polityk RL dla robotów humanoidalnych
- Symulacji fizyki w czasie rzeczywistym
- Transferu Sim-to-Real
- Testowania algorytmów przed wdrożeniem na sprzęcie

## Architektura

### Główne Komponenty

```
┌─────────────────────────────────────┐
│        Isaac Lab Framework          │
├─────────────────────────────────────┤
│  RL Algorithms (PPO, SAC, A3C)     │
│  Task Definitions                   │
│  Reward Functions                   │
├─────────────────────────────────────┤
│     NVIDIA Omniverse Kit            │
│     PhysX 5 (GPU Acceleration)      │
└─────────────────────────────────────┘
         ↕ Python API
┌─────────────────────────────────────┐
│      Your Robot Application         │
└─────────────────────────────────────┘
```

### Kluczowe Cechy

1. **GPU-Accelerated Physics** - tysiące równoległych symulacji
2. **Photo-realistic Rendering** - RTX ray tracing
3. **ROS2 Integration** - bezpośrednia integracja z ROS2
4. **Domain Randomization** - różnorodność środowisk treningowych
5. **Sim-to-Real Transfer** - minimalizacja reality gap

## Instalacja

### Wymagania
- Ubuntu 20.04 lub 22.04
- NVIDIA GPU (RTX 3000+, A100, H100)
- CUDA 11.8+
- Python 3.8+

### Instalacja Krok po Kroku

```bash
# 1. Klonuj repozytorium
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 2. Utwórz środowisko wirtualne
./isaaclab.sh --install

# 3. Aktywuj środowisko
source _isaac_sim/setup_conda_env.sh

# 4. Weryfikacja
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

## Podstawowy Przykład

### Stworzenie Prostego Środowiska

```python
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

# Konfiguracja symulacji
sim_cfg = SimulationCfg(
    dt=0.01,  # 100 Hz
    substeps=1,
    gravity=(0.0, 0.0, -9.81),
    use_gpu_pipeline=True,
    device="cuda:0"
)

# Tworzenie kontekstu symulacji
sim = SimulationContext(sim_cfg)

# Uruchomienie symulacji
sim.reset()
for _ in range(1000):
    sim.step()
```

### Definiowanie Zadania RL

```python
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg, ActionTermCfg
from isaaclab.utils import configclass

@configclass
class HumanoidTaskCfg(ManagerBasedRLEnvCfg):
    """Konfiguracja zadania dla robota humanoidalnego"""
    
    # Definicja robota
    robot_cfg = RobotCfg(
        usd_path="path/to/unitree_g1.usd",
        spawn=sim_utils.UsdFileCfg(
            usd_path="path/to/unitree_g1.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
        ),
    )
    
    # Przestrzeń obserwacji
    observation_space = ObservationGroupCfg(
        policy={
            "joint_pos": ObservationTermCfg(func=get_joint_positions),
            "joint_vel": ObservationTermCfg(func=get_joint_velocities),
            "base_orientation": ObservationTermCfg(func=get_base_orientation),
        }
    )
    
    # Przestrzeń akcji
    action_space = ActionTermCfg(
        joint_efforts=JointEffortActionCfg(
            asset_name="robot",
            joint_names=[".*"],  # Wszystkie stawy
        )
    )
    
    # Funkcja nagrody
    rewards = {
        "forward_velocity": RewardTermCfg(
            func=forward_velocity_reward,
            weight=1.0
        ),
        "energy_penalty": RewardTermCfg(
            func=energy_penalty,
            weight=-0.01
        ),
        "upright_posture": RewardTermCfg(
            func=upright_reward,
            weight=0.5
        ),
    }

# Tworzenie środowiska
env = ManagerBasedRLEnv(cfg=HumanoidTaskCfg())
```

## Trenowanie z RL Algorytmami

### Stable-Baselines3 Integration

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Wrapper dla Isaac Lab env
class IsaacLabWrapper:
    def __init__(self, env):
        self.env = env
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs["policy"]
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs["policy"], reward, done, info

# Trenowanie
env = IsaacLabWrapper(env)
model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    verbose=1,
    device="cuda"
)

# Trening przez 1M kroków
model.learn(total_timesteps=1_000_000)

# Zapis modelu
model.save("unitree_g1_walking")
```

### RSL-RL (ETH Zurich)

```python
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnv

# Konfiguracja algorytmu
cfg_train = {
    'algorithm': {
        'value_loss_coef': 1.0,
        'use_clipped_value_loss': True,
        'clip_param': 0.2,
        'entropy_coef': 0.01,
        'num_learning_epochs': 5,
        'num_mini_batches': 4,
        'learning_rate': 1.e-3,
        'schedule': 'adaptive',
        'gamma': 0.99,
        'lam': 0.95,
        'desired_kl': 0.01,
        'max_grad_norm': 1.0
    }
}

# Runner
runner = OnPolicyRunner(env, cfg_train, log_dir="logs")
runner.learn(num_learning_iterations=1000, init_at_random_ep_len=True)
```

## Domain Randomization

Kluczowa technika dla Sim-to-Real:

```python
from isaaclab.utils.math import sample_uniform

class DomainRandomization:
    def randomize_physics(self, env):
        # Randomizacja masy
        mass_scale = sample_uniform(0.8, 1.2, env.num_envs, device=env.device)
        env.robot.root_physx_view.set_masses(
            env.robot.data.default_mass * mass_scale
        )
        
        # Randomizacja tarcia
        friction = sample_uniform(0.5, 1.5, env.num_envs, device=env.device)
        env.robot.root_physx_view.set_material_properties(
            static_friction=friction,
            dynamic_friction=friction * 0.8
        )
        
        # Randomizacja oświetlenia
        light_intensity = sample_uniform(800, 1200, device=env.device)
        env.scene.update_light_intensity(light_intensity)
```

## Sim-to-Real Transfer

### Pipeline Transferu

1. **Trening w Symulacji**
   - Użyj domain randomization
   - Trenuj na różnorodnych środowiskach
   - Dodaj szum do obserwacji

2. **Walidacja w Symulacji**
   - Test na nowych środowiskach
   - Symulacja opóźnień (latency)
   - Symulacja niedoskonałości sensorów

3. **Transfer na Robota**
   - Kalibracja sensorów
   - Normalizacja obserwacji
   - Fine-tuning na rzeczywistych danych

### Przykład Transferu

```python
class SimToRealBridge:
    def __init__(self, policy_path):
        self.policy = torch.load(policy_path)
        self.obs_normalizer = ObservationNormalizer()
        
    def process_real_observations(self, real_obs):
        # Dodaj szum jak w symulacji
        noisy_obs = real_obs + np.random.normal(0, 0.01, real_obs.shape)
        
        # Normalizacja
        normalized_obs = self.obs_normalizer.normalize(noisy_obs)
        
        return normalized_obs
    
    def get_action(self, obs):
        with torch.no_grad():
            action = self.policy(obs)
        return action.cpu().numpy()
```

## Użycie w Laboratorium

### Unitree G1 Simulation

```python
# Konfiguracja Unitree G1
from isaaclab_assets.unitree import UNITREE_G1_CFG

# Zadanie chodzenia
class UnitreeWalkingEnv(ManagerBasedRLEnv):
    cfg: UnitreeWalkingCfg
    
    def __init__(self, cfg: UnitreeWalkingCfg):
        super().__init__(cfg)
        
        # Inicjalizacja targetów
        self.target_velocity = torch.tensor([1.0, 0.0], device=self.device)
    
    def _get_rewards(self):
        # Nagroda za prędkość do przodu
        velocity_reward = torch.exp(
            -torch.square(
                self.robot.data.root_lin_vel_b[:, 0] - self.target_velocity[0]
            )
        )
        
        # Kara za zużycie energii
        energy_penalty = torch.sum(
            torch.square(self.robot.data.applied_torque), dim=1
        )
        
        return velocity_reward - 0.01 * energy_penalty
```

## Monitorowanie Treningu

### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/unitree_g1_experiment')

for iteration in range(num_iterations):
    # Trening...
    
    # Logowanie metryk
    writer.add_scalar('Reward/mean', mean_reward, iteration)
    writer.add_scalar('Reward/std', std_reward, iteration)
    writer.add_scalar('Policy/entropy', entropy, iteration)
    writer.add_scalar('Loss/value', value_loss, iteration)
```

### Wizualizacja w czasie rzeczywistym

```bash
# Uruchom TensorBoard
tensorboard --logdir=runs/

# Otwórz w przeglądarce
http://localhost:6006
```

## Zasoby

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab)
- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/)
- [PhysX Documentation](https://nvidia-omniverse.github.io/PhysX/)

## Powiązane Artykuły

- [Transfer Sim-to-Real](#wiki-sim-to-real)
- [Uczenie przez Wzmacnianie](#wiki-reinforcement-learning)
- [ROS2 Integration](#wiki-ros2)

---

*Ostatnia aktualizacja: 2025-02-10*  
*Autor: Zespół Akcji, Laboratorium Robotów Humanoidalnych*
