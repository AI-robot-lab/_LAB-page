# Transfer Learning

## Wprowadzenie

**Transfer Learning** to technika uczenia maszynowego, w której wiedza zdobyta podczas rozwiązywania jednego zadania jest wykorzystywana do rozwiązania innego, powiązanego zadania. W robotyce humanoidalnej pozwala na efektywne trenowanie modeli z ograniczoną ilością danych.

## Podstawy Transfer Learning

### Feature Extraction

```python
import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor:
    def __init__(self, model_name='resnet50', pretrained=True):
        # Załaduj pre-trenowany model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            # Usuń ostatnią warstwę (classifier)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.model.eval()
    
    def extract_features(self, images):
        """
        Ekstraktuj features z obrazów
        
        Args:
            images: tensor (batch_size, 3, H, W)
        
        Returns:
            features: tensor (batch_size, feature_dim)
        """
        with torch.no_grad():
            features = self.model(images)
            features = features.flatten(1)
        
        return features

# Użycie
extractor = FeatureExtractor('resnet50')
images = torch.randn(10, 3, 224, 224)
features = extractor.extract_features(images)
print(f"Feature shape: {features.shape}")  # (10, 2048)
```

### Fine-tuning

```python
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, freeze_layers=True):
        super().__init__()
        
        # Załaduj pre-trenowany ResNet
        self.backbone = models.resnet50(pretrained=True)
        
        # Zamroź warstwy backbone (opcjonalnie)
        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Zmień ostatnią warstwę
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_layers(self, num_layers=10):
        """
        Odmroź ostatnie N warstw
        """
        # Pobierz wszystkie parametry
        params = list(self.backbone.parameters())
        
        # Odmroź ostatnie num_layers
        for param in params[-num_layers:]:
            param.requires_grad = True

# Przykład użycia
model = TransferLearningModel(num_classes=10, freeze_layers=True)

# Trening z zamrożonymi warstwami
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

# Po pewnym czasie odmroź warstwy
model.unfreeze_layers(num_layers=20)
```

## Progressive Fine-tuning

```python
class ProgressiveFineTuner:
    def __init__(self, model, num_stages=3):
        self.model = model
        self.num_stages = num_stages
        
        # Podziel warstwy na stages
        self.layer_groups = self.split_layers()
    
    def split_layers(self):
        """
        Podziel model na grupy warstw
        """
        all_layers = list(self.model.backbone.children())
        n = len(all_layers)
        
        layers_per_stage = n // self.num_stages
        
        groups = []
        for i in range(self.num_stages):
            start = i * layers_per_stage
            end = start + layers_per_stage if i < self.num_stages - 1 else n
            groups.append(all_layers[start:end])
        
        return groups
    
    def train_stage(self, stage, train_loader, epochs=10):
        """
        Trenuj jeden stage
        """
        # Odmroź warstwy dla tego stage
        for i in range(stage + 1):
            for layer in self.layer_groups[i]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Trenuj
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001 / (stage + 1)  # Zmniejszaj learning rate
        )
        
        for epoch in range(epochs):
            for batch in train_loader:
                # Training step
                pass
```

## Domain Adaptation

### Adversarial Domain Adaptation

```python
class DomainAdaptationModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        
        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            GradientReversal(alpha=1.0),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # Source vs Target
        )
    
    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        
        # Task prediction
        task_output = self.task_classifier(features)
        
        # Domain prediction (with gradient reversal)
        domain_output = self.domain_classifier(features)
        
        return task_output, domain_output

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
```

## Few-Shot Learning

### Prototypical Networks

```python
class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, support, query, n_way, n_support):
        """
        Args:
            support: support set (n_way * n_support, C, H, W)
            query: query set (n_query, C, H, W)
        """
        # Encode
        z_support = self.encoder(support)
        z_query = self.encoder(query)
        
        # Compute prototypes
        z_support = z_support.view(n_way, n_support, -1)
        prototypes = z_support.mean(dim=1)  # (n_way, feature_dim)
        
        # Compute distances
        distances = self.euclidean_distance(z_query, prototypes)
        
        # Softmax over distances
        log_p_y = nn.functional.log_softmax(-distances, dim=1)
        
        return log_p_y
    
    def euclidean_distance(self, x, y):
        """
        x: (n, d)
        y: (m, d)
        returns: (n, m)
        """
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        
        return torch.pow(x - y, 2).sum(2)
```

## Transfer Learning w Robotyce

### Sim-to-Real Transfer

```python
class SimToRealTransfer:
    def __init__(self, sim_model):
        self.sim_model = sim_model
        
        # Domain randomization parameters
        self.randomization_params = {
            'lighting': {'min': 0.5, 'max': 1.5},
            'texture': {'variations': 10},
            'physics': {'friction': (0.5, 1.5)}
        }
    
    def apply_domain_randomization(self, sim_env):
        """
        Zastosuj domain randomization w symulacji
        """
        # Randomizuj oświetlenie
        lighting_scale = np.random.uniform(
            self.randomization_params['lighting']['min'],
            self.randomization_params['lighting']['max']
        )
        sim_env.set_lighting(lighting_scale)
        
        # Randomizuj tekstury
        texture_id = np.random.randint(
            self.randomization_params['texture']['variations']
        )
        sim_env.set_texture(texture_id)
        
        # Randomizuj fizykę
        friction = np.random.uniform(
            *self.randomization_params['physics']['friction']
        )
        sim_env.set_friction(friction)
    
    def train_with_randomization(self, num_episodes=1000):
        """
        Trenuj z domain randomization
        """
        for episode in range(num_episodes):
            # Randomizuj środowisko
            self.apply_domain_randomization(self.sim_env)
            
            # Trenuj
            self.train_episode()
```

## Meta-Learning

### MAML (Model-Agnostic Meta-Learning)

```python
class MAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=outer_lr
        )
    
    def inner_loop(self, task_data, num_steps=5):
        """
        Adaptacja do nowego zadania (inner loop)
        """
        # Sklonuj model
        adapted_params = [p.clone() for p in self.model.parameters()]
        
        for step in range(num_steps):
            # Forward
            loss = self.compute_loss(task_data, adapted_params)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_params,
                create_graph=True
            )
            
            # Update
            adapted_params = [
                p - self.inner_lr * g
                for p, g in zip(adapted_params, grads)
            ]
        
        return adapted_params
    
    def outer_loop(self, tasks):
        """
        Meta-update (outer loop)
        """
        meta_loss = 0
        
        for task in tasks:
            # Inner loop adaptation
            adapted_params = self.inner_loop(task['train'])
            
            # Compute loss on query set
            query_loss = self.compute_loss(task['query'], adapted_params)
            meta_loss += query_loss
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
```

## Powiązane Artykuły

- [Deep Learning](#wiki-deep-learning)
- [Neural Networks](#wiki-neural-networks)
- [Reinforcement Learning](#wiki-reinforcement-learning)
- [Sim-to-Real](#wiki-sim-to-real)

---

*Ostatnia aktualizacja: 2025-02-12*  
*Autor: Zespół Kognicji, Laboratorium Robotów Humanoidalnych*
