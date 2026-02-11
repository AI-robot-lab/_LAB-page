# Uczenie przez Wzmacnianie (Reinforcement Learning)

## Wprowadzenie

**Reinforcement Learning (RL)** to paradygmat uczenia maszynowego, w którym agent uczy się optymalnego zachowania poprzez interakcję ze środowiskiem. W robotyce humanoidalnej RL jest kluczowe do uczenia złożonych zachowań motorycznych i podejmowania decyzji.

## Podstawowe Pojęcia

### Markov Decision Process (MDP)

```
MDP = (S, A, P, R, γ)

S - Zbiór stanów (states)
A - Zbiór akcji (actions)
P - Funkcja przejścia P(s'|s,a)
R - Funkcja nagrody R(s,a,s')
γ - Współczynnik dyskontowania (discount factor)
```

```python
import numpy as np

class SimpleMDP:
    def __init__(self, n_states=5, n_actions=2):
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Random transition probabilities
        self.P = np.random.rand(n_states, n_actions, n_states)
        self.P = self.P / self.P.sum(axis=2, keepdims=True)
        
        # Random rewards
        self.R = np.random.randn(n_states, n_actions, n_states)
        
        self.gamma = 0.99
        self.state = 0
    
    def step(self, action):
        """
        Wykonaj akcję i przejdź do nowego stanu
        """
        # Sample next state
        next_state = np.random.choice(
            self.n_states,
            p=self.P[self.state, action]
        )
        
        # Get reward
        reward = self.R[self.state, action, next_state]
        
        self.state = next_state
        
        return next_state, reward, False  # state, reward, done
```

## Q-Learning

### Tabular Q-Learning

```python
class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        self.n_states = n_states
        self.n_actions = n_actions
    
    def select_action(self, state):
        """
        Epsilon-greedy action selection
        """
        if np.random.rand() < self.epsilon:
            # Explore
            return np.random.randint(self.n_actions)
        else:
            # Exploit
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state):
        """
        Q-Learning update rule
        """
        # Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        
        self.Q[state, action] += self.alpha * td_error
        
        return td_error

# Training loop
env = SimpleMDP(n_states=10, n_actions=4)
agent = QLearning(n_states=10, n_actions=4)

for episode in range(1000):
    state = env.reset()
    total_reward = 0
    
    for step in range(100):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        
        agent.update(state, action, reward, next_state)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.2f}")
```

### Deep Q-Network (DQN)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
```

## Policy Gradient Methods

### REINFORCE Algorithm

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.net(state)

class REINFORCE:
    def __init__(self, state_dim, action_dim):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.gamma = 0.99
        
        self.saved_log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        
        # Sample action from probability distribution
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        # Save log probability
        self.saved_log_probs.append(m.log_prob(action))
        
        return action.item()
    
    def update(self):
        """
        Update policy after episode
        """
        # Calculate returns
        returns = []
        G = 0
        
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        self.saved_log_probs = []
        self.rewards = []
        
        return policy_loss.item()
```

## Actor-Critic

### A2C (Advantage Actor-Critic)

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        features = self.shared(state)
        policy = self.actor(features)
        value = self.critic(features)
        return policy, value

class A2C:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        policy, value = self.model(state)
        
        m = torch.distributions.Categorical(policy)
        action = m.sample()
        
        return action.item(), m.log_prob(action), value
    
    def update(self, log_prob, value, reward, next_value, done):
        """
        A2C update
        """
        # TD error (advantage)
        advantage = reward + self.gamma * next_value * (1 - done) - value
        
        # Actor loss
        actor_loss = -log_prob * advantage.detach()
        
        # Critic loss
        critic_loss = advantage.pow(2)
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

## PPO (Proximal Policy Optimization)

```python
class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.gamma = 0.99
        self.epsilon = 0.2  # Clipping parameter
        self.epochs = 10
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.actor(state)
        
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        return action.item(), m.log_prob(action)
    
    def update(self, states, actions, old_log_probs, rewards, dones):
        """
        PPO update with clipped objective
        """
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # Calculate returns and advantages
        returns = self.compute_returns(rewards, dones)
        values = self.critic(states).squeeze()
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.epochs):
            # Actor update
            probs = self.actor(states)
            m = torch.distributions.Categorical(probs)
            new_log_probs = m.log_prob(actions)
            
            # Probability ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Critic update
            values = self.critic(states).squeeze()
            critic_loss = nn.MSELoss()(values, returns)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
    
    def compute_returns(self, rewards, dones):
        returns = []
        G = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + self.gamma * G * (1 - done)
            returns.insert(0, G)
        
        return torch.tensor(returns)
```

## RL dla Robotyki

### Aplikacja: Locomotion

```python
import isaacgym
from isaacgymenvs.tasks import Humanoid

class HumanoidLocomotionRL:
    def __init__(self):
        # Isaac Gym environment
        self.env = Humanoid(cfg, sim_device, graphics_device_id, headless)
        
        # PPO agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.agent = PPO(state_dim, action_dim)
    
    def train(self, num_episodes=10000):
        for episode in range(num_episodes):
            states, actions, log_probs, rewards, dones = [], [], [], [], []
            
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(1000):
                # Select action
                action, log_prob = self.agent.select_action(state)
                
                # Step environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob.item())
                rewards.append(reward)
                dones.append(done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Update policy
            self.agent.update(states, actions, log_probs, rewards, dones)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}")
```

## Porównanie Algorytmów

| Algorytm | Typ | Sample Efficiency | Stabilność | Use Case |
|----------|-----|-------------------|------------|----------|
| **Q-Learning** | Value-based | Niska | Wysoka | Dyskretne akcje |
| **DQN** | Value-based | Średnia | Średnia | Atari, gry |
| **REINFORCE** | Policy-based | Niska | Niska | Proste zadania |
| **A2C** | Actor-Critic | Średnia | Średnia | Continuous control |
| **PPO** | Actor-Critic | Wysoka | Wysoka | Robotyka, locomotion |

## Powiązane Artykuły

- [Deep Learning](#wiki-deep-learning)
- [Isaac Lab](#wiki-isaac-lab) - Symulacja dla RL
- [Neural Networks](#wiki-neural-networks)
- [PyTorch](#wiki-pytorch)

## Zasoby

- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [RLlib](https://docs.ray.io/en/latest/rllib/index.html)

---

*Ostatnia aktualizacja: 2025-02-10*  
*Autor: Zespół Kognicji, Laboratorium Robotów Humanoidalnych*
