# Sim-to-Real Transfer

## Wprowadzenie

**Sim-to-Real Transfer** to proces przenoszenia polityk i modeli wytrenowanych w symulacji do rzeczywistych robotów. W robotyce humanoidalnej umożliwia bezpieczne i efektywne uczenie skomplikowanych zachowań.

## Reality Gap Problem

### Źródła Różnic

```python
class RealityGap:
    def __init__(self):
        self.gap_sources = {
            'physics': {
                'friction': 'Niedokładne modelowanie tarcia',
                'contact': 'Uproszczone modele kontaktu',
                'dynamics': 'Nieznane parametry dynamiczne'
            },
            'sensors': {
                'noise': 'Idealny vs rzeczywisty szum',
                'latency': 'Opóźnienia w rzeczywistości',
                'resolution': 'Ograniczona rozdzielczość'
            },
            'actuators': {
                'delays': 'Opóźnienia w wykonaniu',
                'backlash': 'Luz w przekładniach',
                'saturation': 'Limity prędkości/momentu'
            }
        }
    
    def measure_gap(self, sim_trajectory, real_trajectory):
        """
        Zmierz różnicę między symulacją a rzeczywistością
        """
        import numpy as np
        
        # Błąd pozycji
        pos_error = np.mean(np.abs(sim_trajectory - real_trajectory))
        
        # Błąd prędkości
        sim_vel = np.diff(sim_trajectory, axis=0)
        real_vel = np.diff(real_trajectory, axis=0)
        vel_error = np.mean(np.abs(sim_vel - real_vel))
        
        return {
            'position_error': pos_error,
            'velocity_error': vel_error,
            'max_deviation': np.max(np.abs(sim_trajectory - real_trajectory))
        }
```

## Domain Randomization

### Podstawowa Randomizacja

```python
class DomainRandomizer:
    def __init__(self, env):
        self.env = env
        
        # Zakresy randomizacji
        self.params = {
            'mass': (0.8, 1.2),           # ±20%
            'friction': (0.5, 1.5),       # ±50%
            'damping': (0.7, 1.3),        # ±30%
            'motor_strength': (0.9, 1.1), # ±10%
            'latency': (0.0, 0.05),       # 0-50ms
            'lighting': (0.5, 2.0),       # Intensywność
            'camera_noise': (0.0, 0.05)   # Szum sensora
        }
    
    def randomize_physics(self):
        """
        Randomizuj parametry fizyki
        """
        import numpy as np
        
        for link in self.env.robot.links:
            # Masa
            mass_scale = np.random.uniform(*self.params['mass'])
            link.mass *= mass_scale
            
            # Tarcie
            friction = np.random.uniform(*self.params['friction'])
            link.friction = friction
            
            # Tłumienie
            damping = np.random.uniform(*self.params['damping'])
            link.damping = damping
    
    def randomize_actuators(self):
        """
        Randomizuj charakterystyki aktuatorów
        """
        import numpy as np
        
        for motor in self.env.robot.motors:
            # Siła silnika
            strength = np.random.uniform(*self.params['motor_strength'])
            motor.max_force *= strength
            
            # Opóźnienie
            latency = np.random.uniform(*self.params['latency'])
            motor.latency = latency
    
    def randomize_sensors(self):
        """
        Randomizuj sensory
        """
        import numpy as np
        
        # Szum kamery
        noise_level = np.random.uniform(*self.params['camera_noise'])
        self.env.camera.noise_stddev = noise_level
        
        # Oświetlenie
        lighting = np.random.uniform(*self.params['lighting'])
        self.env.scene.lighting_intensity = lighting
```

### Automatyczna Randomizacja (ADR)

```python
class AutomaticDomainRandomization:
    def __init__(self, initial_ranges):
        self.param_ranges = initial_ranges
        self.performance_history = []
        
        # Progi adaptacji
        self.success_threshold = 0.8
        self.expand_rate = 1.1
        self.contract_rate = 0.9
    
    def update_ranges(self, performance):
        """
        Automatycznie dostosuj zakresy randomizacji
        """
        self.performance_history.append(performance)
        
        # Sprawdź ostatnie N epizodów
        recent_performance = self.performance_history[-100:]
        success_rate = sum(recent_performance) / len(recent_performance)
        
        if success_rate > self.success_threshold:
            # Rozszerz zakresy - zadanie za łatwe
            for param in self.param_ranges:
                low, high = self.param_ranges[param]
                center = (low + high) / 2
                range_size = (high - low) / 2
                
                new_range = range_size * self.expand_rate
                self.param_ranges[param] = (
                    center - new_range,
                    center + new_range
                )
        else:
            # Zmniejsz zakresy - zadanie za trudne
            for param in self.param_ranges:
                low, high = self.param_ranges[param]
                center = (low + high) / 2
                range_size = (high - low) / 2
                
                new_range = range_size * self.contract_rate
                self.param_ranges[param] = (
                    center - new_range,
                    center + new_range
                )
        
        return self.param_ranges
```

## System Identification

### Parameter Estimation

```python
class SystemIdentification:
    def __init__(self, robot_model):
        self.model = robot_model
        self.estimated_params = {}
    
    def collect_data(self, num_trajectories=100):
        """
        Zbierz dane z rzeczywistego robota
        """
        data = []
        
        for _ in range(num_trajectories):
            # Wykonaj losową trajektorię
            trajectory = self.execute_random_trajectory()
            
            # Zapisz: (states, actions, next_states)
            data.append(trajectory)
        
        return data
    
    def estimate_parameters(self, data):
        """
        Estymuj parametry modelu na podstawie danych
        """
        from scipy.optimize import minimize
        
        def loss_function(params):
            # Ustaw parametry w modelu
            self.model.set_parameters(params)
            
            # Oblicz błąd predykcji
            total_error = 0
            for states, actions, next_states in data:
                predicted = self.model.predict(states, actions)
                error = np.mean((predicted - next_states)**2)
                total_error += error
            
            return total_error
        
        # Optymalizuj parametry
        initial_params = self.model.get_parameters()
        result = minimize(loss_function, initial_params, method='L-BFGS-B')
        
        self.estimated_params = result.x
        return self.estimated_params
```

## Progressive Training

### Curriculum Learning

```python
class ProgressiveSimToReal:
    def __init__(self, easy_env, hard_env):
        self.easy_env = easy_env
        self.hard_env = hard_env
        
        # Etapy treningu
        self.stages = [
            {'name': 'ideal', 'randomization': 0.0, 'episodes': 1000},
            {'name': 'light', 'randomization': 0.3, 'episodes': 2000},
            {'name': 'medium', 'randomization': 0.6, 'episodes': 3000},
            {'name': 'heavy', 'randomization': 1.0, 'episodes': 5000}
        ]
    
    def train(self, policy):
        """
        Progresywny trening od łatwego do trudnego
        """
        for stage in self.stages:
            print(f"Stage: {stage['name']}")
            
            # Ustaw poziom randomizacji
            self.set_randomization_level(stage['randomization'])
            
            # Trenuj
            for episode in range(stage['episodes']):
                self.train_episode(policy)
            
            # Ewaluacja
            success_rate = self.evaluate(policy)
            print(f"Success rate: {success_rate:.2f}")
            
            # Jeśli za słabo, powtórz stage
            if success_rate < 0.7:
                print("Repeating stage...")
                continue
    
    def set_randomization_level(self, level):
        """
        Ustaw poziom randomizacji środowiska
        """
        self.env.randomizer.set_intensity(level)
```

## Residual Learning

```python
class ResidualPolicy:
    def __init__(self, sim_policy, residual_network):
        self.sim_policy = sim_policy  # Polityka z symulacji
        self.residual_network = residual_network  # Korekta
    
    def forward(self, observation):
        """
        Akcja = polityka_sim + rezydualna_korekta
        """
        # Bazowa akcja z symulacji
        base_action = self.sim_policy(observation)
        
        # Korekta dla rzeczywistości
        residual = self.residual_network(observation)
        
        # Połącz
        final_action = base_action + residual
        
        return final_action
    
    def train_residual(self, real_data):
        """
        Trenuj tylko sieć rezydualną na rzeczywistych danych
        """
        # Zamroź politykę z symulacji
        for param in self.sim_policy.parameters():
            param.requires_grad = False
        
        # Trenuj tylko residual
        for observation, expert_action in real_data:
            base_action = self.sim_policy(observation)
            predicted_residual = self.residual_network(observation)
            
            # Loss: różnica między ekspertem a (base + residual)
            loss = (expert_action - (base_action + predicted_residual)).pow(2).mean()
            
            loss.backward()
            self.optimizer.step()
```

## Visual Domain Adaptation

### Image Translation

```python
class VisualDomainAdapter:
    def __init__(self, sim_to_real_gan):
        self.generator = sim_to_real_gan
    
    def adapt_observation(self, sim_image):
        """
        Przekształć obraz z symulacji na realistyczny
        """
        # Normalize
        sim_image = self.normalize(sim_image)
        
        # Generate realistic image
        with torch.no_grad():
            real_like_image = self.generator(sim_image)
        
        return real_like_image
    
    def train_translator(self, sim_images, real_images):
        """
        Trenuj CycleGAN do tłumaczenia sim→real
        """
        # CycleGAN training
        # G: sim → real
        # F: real → sim
        # Cycle consistency: F(G(sim)) ≈ sim
        pass
```

## Validation Strategy

```python
class SimToRealValidator:
    def __init__(self, sim_env, real_robot):
        self.sim_env = sim_env
        self.real_robot = real_robot
    
    def validate_transfer(self, policy, num_trials=50):
        """
        Waliduj transfer na rzeczywistym robocie
        """
        sim_results = []
        real_results = []
        
        for trial in range(num_trials):
            # Test w symulacji
            sim_success = self.test_in_sim(policy)
            sim_results.append(sim_success)
            
            # Test w rzeczywistości
            real_success = self.test_in_real(policy)
            real_results.append(real_success)
        
        # Analiza
        sim_success_rate = sum(sim_results) / len(sim_results)
        real_success_rate = sum(real_results) / len(real_results)
        
        transfer_gap = sim_success_rate - real_success_rate
        
        print(f"Sim success rate: {sim_success_rate:.2%}")
        print(f"Real success rate: {real_success_rate:.2%}")
        print(f"Transfer gap: {transfer_gap:.2%}")
        
        return {
            'sim_success': sim_success_rate,
            'real_success': real_success_rate,
            'gap': transfer_gap
        }
```

## Powiązane Artykuły

- [Isaac Lab](#wiki-isaac-lab)
- [Reinforcement Learning](#wiki-reinforcement-learning)
- [Transfer Learning](#wiki-transfer-learning)
- [Manipulation](#wiki-manipulation)

---

*Ostatnia aktualizacja: 2025-02-12*  
*Autor: Zespół Interakcji, Laboratorium Robotów Humanoidalnych*
