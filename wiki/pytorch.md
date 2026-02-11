# PyTorch - Deep Learning Framework

## Wprowadzenie

**PyTorch** to open-source framework do deep learning rozwijany przez Meta AI. Jest podstawowym narzędziem w Laboratorium do trenowania modeli percepcji, kognicji i kontroli robotów.

## Podstawy

### Tensory

```python
import torch

# Tworzenie tensorów
x = torch.tensor([[1, 2], [3, 4]])
y = torch.zeros(2, 3)
z = torch.randn(2, 2)  # Random normal

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_gpu = x.to(device)

# Operacje
a = torch.add(x, z)
b = torch.matmul(x, z)
c = x * z  # Element-wise

# Indexing
print(x[0, 1])  # 2
print(x[:, 0])  # [1, 3]
```

### Autograd

```python
# Gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

# Backward pass
y.backward()
print(x.grad)  # dy/dx = 2x = 4.0

# Multiple operations
x = torch.randn(3, requires_grad=True)
y = x * 2

while y.data.norm() < 1000:
    y = y * 2

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)

print(x.grad)
```

## Tworzenie Modeli

### Sequential API

```python
import torch.nn as nn

# Prosty MLP
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.Softmax(dim=1)
)

# Forward pass
x = torch.randn(32, 784)  # batch of 32
output = model(x)
```

### Custom Models

```python
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Użycie
model = ConvNet(num_classes=10)
```

## Training Loop

```python
import torch.optim as optim

# Model, loss, optimizer
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# Validation
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc

# Full training
num_epochs = 10

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
    print(f'  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
```

## DataLoaders

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset i DataLoader
dataset = CustomDataset(data, labels, transform=transform)
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4
)
```

## Transfer Learning

```python
from torchvision import models

# Załaduj pre-trained model
model = models.resnet50(pretrained=True)

# Freeze wagi
for param in model.parameters():
    param.requires_grad = False

# Zamień ostatnią warstwę
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Trenuj tylko ostatnią warstwę
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

## Model Saving & Loading

```python
# Save całego modelu
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# Save tylko state dict (recommended)
torch.save(model.state_dict(), 'model_weights.pth')

# Load
model = ConvNet()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

## Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for inputs, labels in dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
    
    optimizer.zero_grad()
    
    # Autocast dla forward pass
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
    # Scaled backward
    scaler.scale(loss).backward()
    
    # Unscale i update
    scaler.step(optimizer)
    scaler.update()
```

## Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    # Model na GPU
    model = ConvNet().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Training loop
    for epoch in range(num_epochs):
        # ... training code
        pass
    
    cleanup()

# Launch
import torch.multiprocessing as mp
mp.spawn(train_ddp, args=(world_size,), nprocs=world_size)
```

## Custom Loss Functions

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(
                torch.clamp(self.margin - euclidean_distance, min=0.0), 2
            )
        )
        return loss_contrastive
```

## TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment1')

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = validate(...)
    
    # Log scalars
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    # Log model graph
    if epoch == 0:
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        writer.add_graph(model, dummy_input)

writer.close()

# Visualize: tensorboard --logdir=runs
```

## Robotics Applications

### Vision Model dla Detekcji

```python
class ObjectDetector(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # YOLO-style architecture
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        self.detection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes * 5)  # [x, y, w, h, conf] per class
        )
    
    def forward(self, x):
        features = self.backbone(x)
        detections = self.detection_head(features)
        return detections.view(-1, num_classes, 5)
```

### Policy Network dla RL

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        action = self.policy(state)
        value = self.value(state)
        return action, value
```

## Powiązane Artykuły

- [Deep Learning](#wiki-deep-learning)
- [Uczenie przez Wzmacnianie](#wiki-reinforcement-learning)
- [Computer Vision](#wiki-computer-vision)

## Zasoby

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Examples](https://github.com/pytorch/examples)

---

*Ostatnia aktualizacja: 2025-02-10*  
*Autor: Laboratorium Robotów Humanoidalnych, PRz*
