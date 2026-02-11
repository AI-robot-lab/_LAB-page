# Deep Learning - Głębokie Uczenie

## Wprowadzenie

**Deep Learning** to poddziedzina uczenia maszynowego wykorzystująca głębokie sieci neuronowe do uczenia hierarchicznych reprezentacji danych. Jest fundamentem nowoczesnej percepcji, kognicji i kontroli w robotyce humanoidalnej.

## Fundamenty

### Perceptron Wielowarstwowy (MLP)

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        """
        layers: lista wymiarów warstw, np. [784, 128, 64, 10]
        """
        self.weights = []
        self.biases = []
        
        # Inicjalizacja Xavier
        for i in range(len(layers) - 1):
            W = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            
            self.weights.append(W)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation
        """
        self.activations = [X]
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            
            if i == len(self.weights) - 1:
                # Output layer - softmax
                a = self.softmax(z)
            else:
                # Hidden layers - ReLU
                a = self.relu(z)
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate=0.01):
        """
        Backpropagation
        """
        m = X.shape[0]
        
        # Output layer gradient
        delta = self.activations[-1] - y
        
        # Backpropagate
        for i in reversed(range(len(self.weights))):
            # Gradients
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            if i > 0:
                # Propagate to previous layer
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.activations[i])
    
    def train(self, X, y, epochs=100, learning_rate=0.01):
        """
        Training loop
        """
        for epoch in range(epochs):
            # Forward
            output = self.forward(X)
            
            # Loss
            loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))
            
            # Backward
            self.backward(X, y, learning_rate)
            
            if epoch % 10 == 0:
                accuracy = np.mean(np.argmax(output, axis=1) == np.argmax(y, axis=1))
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
```

## Convolutional Neural Networks (CNN)

### Podstawowa Architektura

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

### ResNet - Residual Networks

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add residual connection
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, 1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
```

## Recurrent Neural Networks (RNN)

### LSTM

```python
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))
        
        # Get last time step
        out = out[:, -1, :]
        
        # Fully connected
        out = self.fc(out)
        
        return out
```

## Regularization Techniques

### Dropout

```python
class DropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(784, 512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Dropout tylko podczas treningu
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x
```

### Batch Normalization

```python
class BatchNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # Normalizacja przed aktywacją
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        return x
```

## Optimization Algorithms

### Adam Optimizer

```python
import torch.optim as optim

model = ResNet()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=1e-5
)

# Learning rate scheduling
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1
)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
    
    scheduler.step()
```

## Transfer Learning

```python
from torchvision import models

# Załaduj pre-trained model
model = models.resnet50(pretrained=True)

# Freeze wszystkie warstwy
for param in model.parameters():
    param.requires_grad = False

# Zamień ostatnią warstwę
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Trenuj tylko ostatnią warstwę
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

## Data Augmentation

```python
from torchvision import transforms

# Augmentacje dla treningu
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Bez augmentacji dla walidacji
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

## Archite ktury dla Robotyki

| Architektura | Zastosowanie | Zalety |
|--------------|-------------|---------|
| **CNN** | Wizja komputerowa | Spatial features |
| **ResNet** | Image classification | Skip connections |
| **LSTM** | Sekwencje czasowe | Long-term memory |
| **Transformer** | NLP, Vision | Attention mechanism |
| **VAE** | Generowanie | Latent space |

## Powiązane Artykuły

- [PyTorch](#wiki-pytorch)
- [Computer Vision](#wiki-computer-vision)
- [Neural Networks](#wiki-neural-networks)
- [Transformers](#wiki-transformers)

---

*Ostatnia aktualizacja: 2025-02-10*  
*Autor: Zespół Kognicji, Laboratorium Robotów Humanoidalnych*
