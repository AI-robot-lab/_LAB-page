# Sieci Neuronowe (Neural Networks)

## Wprowadzenie

**Sieci neuronowe** są podstawą deep learning i sztucznej inteligencji. Zainspirowane biologicznymi neuronami, składają się z połączonych węzłów przetwarzających informacje w warstwach.

## Pojedynczy Neuron

### Model Matematyczny

```python
import numpy as np

class Neuron:
    def __init__(self, n_inputs):
        # Inicjalizacja losowych wag
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()
    
    def forward(self, inputs):
        """
        Forward pass neuronu
        
        output = σ(w·x + b)
        """
        # Suma ważona
        z = np.dot(self.weights, inputs) + self.bias
        
        # Funkcja aktywacji (sigmoid)
        output = self.sigmoid(z)
        
        return output
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
```

## Funkcje Aktywacji

### Popularne Funkcje

```python
import numpy as np
import matplotlib.pyplot as plt

class ActivationFunctions:
    
    @staticmethod
    def sigmoid(x):
        """
        σ(x) = 1 / (1 + e^(-x))
        Zakres: (0, 1)
        """
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        """
        tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        Zakres: (-1, 1)
        """
        return np.tanh(x)
    
    @staticmethod
    def relu(x):
        """
        ReLU(x) = max(0, x)
        Zakres: [0, ∞)
        """
        return np.maximum(0, x)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """
        LeakyReLU(x) = max(αx, x)
        Zapobiega "dying ReLU"
        """
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def elu(x, alpha=1.0):
        """
        ELU(x) = x if x > 0 else α(e^x - 1)
        Smooth w okolicy zera
        """
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def swish(x):
        """
        Swish(x) = x · σ(x)
        Self-gated activation
        """
        return x * (1 / (1 + np.exp(-x)))
    
    @staticmethod
    def softmax(x):
        """
        Softmax dla klasyfikacji multi-class
        Σ(output) = 1
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0, keepdims=True)

# Wizualizacja
x = np.linspace(-5, 5, 100)
activations = ActivationFunctions()

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(x, activations.sigmoid(x))
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(x, activations.tanh(x))
plt.title('Tanh')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(x, activations.relu(x))
plt.title('ReLU')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(x, activations.leaky_relu(x))
plt.title('Leaky ReLU')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(x, activations.elu(x))
plt.title('ELU')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(x, activations.swish(x))
plt.title('Swish')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Feedforward Neural Network

### Implementacja od Zera

```python
class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: lista wymiarów warstw
        np. [784, 128, 64, 10] dla MNIST
        """
        self.layers = []
        self.layer_sizes = layer_sizes
        
        # Inicjalizacja wag (Xavier/He)
        for i in range(len(layer_sizes) - 1):
            layer = {
                'weights': np.random.randn(
                    layer_sizes[i], 
                    layer_sizes[i+1]
                ) * np.sqrt(2.0 / layer_sizes[i]),
                'bias': np.zeros((1, layer_sizes[i+1]))
            }
            self.layers.append(layer)
        
        self.activations = []
        self.z_values = []
    
    def forward(self, X):
        """
        Forward propagation
        """
        self.activations = [X]
        self.z_values = []
        
        A = X
        
        for i, layer in enumerate(self.layers):
            # Linear transformation
            Z = np.dot(A, layer['weights']) + layer['bias']
            self.z_values.append(Z)
            
            # Activation
            if i == len(self.layers) - 1:
                # Output layer - softmax
                A = self.softmax(Z)
            else:
                # Hidden layers - ReLU
                A = self.relu(Z)
            
            self.activations.append(A)
        
        return A
    
    def backward(self, X, y, learning_rate=0.01):
        """
        Backpropagation
        """
        m = X.shape[0]
        
        # Output layer gradient
        dZ = self.activations[-1] - y
        
        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            # Gradients
            dW = (1/m) * np.dot(self.activations[i].T, dZ)
            db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            # Update weights
            self.layers[i]['weights'] -= learning_rate * dW
            self.layers[i]['bias'] -= learning_rate * db
            
            if i > 0:
                # Propagate gradient to previous layer
                dA = np.dot(dZ, self.layers[i]['weights'].T)
                dZ = dA * self.relu_derivative(self.z_values[i-1])
    
    def train(self, X, y, epochs=100, batch_size=32, learning_rate=0.01):
        """
        Training loop z mini-batch gradient descent
        """
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward
                output = self.forward(X_batch)
                
                # Backward
                self.backward(X_batch, y_batch, learning_rate)
            
            # Calculate loss
            if epoch % 10 == 0:
                output = self.forward(X)
                loss = self.cross_entropy_loss(y, output)
                accuracy = np.mean(np.argmax(output, axis=1) == np.argmax(y, axis=1))
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    def predict(self, X):
        """
        Prediction
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    # Utility functions
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m
```

## Inicjalizacja Wag

### Metody Inicjalizacji

```python
class WeightInitialization:
    
    @staticmethod
    def zeros(shape):
        """
        Zero initialization (NIE UŻYWAĆ!)
        Problem: wszystkie neurony uczą się tego samego
        """
        return np.zeros(shape)
    
    @staticmethod
    def random_uniform(shape, low=-0.1, high=0.1):
        """
        Uniform random initialization
        """
        return np.random.uniform(low, high, shape)
    
    @staticmethod
    def xavier(shape):
        """
        Xavier/Glorot initialization
        Dla sigmoid/tanh
        
        Var(W) = 1/n_in
        """
        n_in = shape[0]
        return np.random.randn(*shape) * np.sqrt(1.0 / n_in)
    
    @staticmethod
    def he(shape):
        """
        He initialization
        Dla ReLU
        
        Var(W) = 2/n_in
        """
        n_in = shape[0]
        return np.random.randn(*shape) * np.sqrt(2.0 / n_in)
    
    @staticmethod
    def lecun(shape):
        """
        LeCun initialization
        Dla SELU
        
        Var(W) = 1/n_in
        """
        n_in = shape[0]
        return np.random.randn(*shape) * np.sqrt(1.0 / n_in)
```

## Regularization Techniques

### L1 i L2 Regularization

```python
class RegularizedNetwork(NeuralNetwork):
    
    def __init__(self, layer_sizes, l1_lambda=0.0, l2_lambda=0.0):
        super().__init__(layer_sizes)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
    
    def compute_loss(self, y_true, y_pred):
        """
        Loss z regularizacją
        """
        # Cross-entropy
        ce_loss = self.cross_entropy_loss(y_true, y_pred)
        
        # L1 regularization (Lasso)
        l1_loss = 0
        if self.l1_lambda > 0:
            for layer in self.layers:
                l1_loss += np.sum(np.abs(layer['weights']))
            l1_loss *= self.l1_lambda
        
        # L2 regularization (Ridge)
        l2_loss = 0
        if self.l2_lambda > 0:
            for layer in self.layers:
                l2_loss += np.sum(layer['weights'] ** 2)
            l2_loss *= (self.l2_lambda / 2)
        
        return ce_loss + l1_loss + l2_loss
    
    def backward(self, X, y, learning_rate=0.01):
        """
        Backpropagation z regularizacją
        """
        m = X.shape[0]
        dZ = self.activations[-1] - y
        
        for i in reversed(range(len(self.layers))):
            dW = (1/m) * np.dot(self.activations[i].T, dZ)
            db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            # Add regularization gradient
            if self.l1_lambda > 0:
                dW += self.l1_lambda * np.sign(self.layers[i]['weights'])
            
            if self.l2_lambda > 0:
                dW += self.l2_lambda * self.layers[i]['weights']
            
            # Update
            self.layers[i]['weights'] -= learning_rate * dW
            self.layers[i]['bias'] -= learning_rate * db
            
            if i > 0:
                dA = np.dot(dZ, self.layers[i]['weights'].T)
                dZ = dA * self.relu_derivative(self.z_values[i-1])
```

### Dropout

```python
class DropoutLayer:
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob
        self.mask = None
    
    def forward(self, X, training=True):
        """
        Forward pass z dropout
        """
        if training:
            # Create dropout mask
            self.mask = np.random.rand(*X.shape) < self.keep_prob
            
            # Apply mask and scale
            return X * self.mask / self.keep_prob
        else:
            # No dropout during inference
            return X
    
    def backward(self, dA):
        """
        Backward pass
        """
        return dA * self.mask / self.keep_prob
```

## Batch Normalization

```python
class BatchNormLayer:
    def __init__(self, n_features, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)
        
        # Running statistics
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
        
        # Cache for backprop
        self.cache = None
    
    def forward(self, X, training=True):
        """
        Forward pass
        """
        if training:
            # Batch statistics
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            
            # Normalize
            X_norm = (X - mean) / np.sqrt(var + self.epsilon)
            
            # Scale and shift
            out = self.gamma * X_norm + self.beta
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            # Cache for backprop
            self.cache = (X, X_norm, mean, var)
            
            return out
        else:
            # Use running statistics
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * X_norm + self.beta
    
    def backward(self, dout):
        """
        Backward pass
        """
        X, X_norm, mean, var = self.cache
        m = X.shape[0]
        
        # Gradients
        dgamma = np.sum(dout * X_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
        
        dX_norm = dout * self.gamma
        
        dvar = np.sum(dX_norm * (X - mean) * -0.5 * (var + self.epsilon)**(-1.5), axis=0)
        dmean = np.sum(dX_norm * -1 / np.sqrt(var + self.epsilon), axis=0) + \
                dvar * np.sum(-2 * (X - mean), axis=0) / m
        
        dX = dX_norm / np.sqrt(var + self.epsilon) + \
             dvar * 2 * (X - mean) / m + \
             dmean / m
        
        return dX, dgamma, dbeta
```

## Porównanie Architektur

| Architektura | Warstwy | Parametry | Zastosowanie |
|--------------|---------|-----------|--------------|
| **Perceptron** | 1 | ~1K | Klasyfikacja binarna |
| **MLP** | 3-5 | ~100K | Tabular data |
| **Deep NN** | 10+ | ~1M | Complex patterns |
| **CNN** | 20+ | ~10M | Images |
| **RNN/LSTM** | 5-10 | ~1M | Sequences |
| **Transformer** | 12-96 | ~100M-1B | NLP, Vision |

## Powiązane Artykuły

- [Deep Learning](#wiki-deep-learning)
- [PyTorch](#wiki-pytorch)
- [Reinforcement Learning](#wiki-reinforcement-learning)

## Zasoby

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

---

*Ostatnia aktualizacja: 2025-02-11*  
*Autor: Zespół Kognicji, Laboratorium Robotów Humanoidalnych*
