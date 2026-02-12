# Architektury Transformer

## Wprowadzenie

**Transformer** to architektura sieci neuronowej oparta na mechanizmie uwagi (attention), która zrewolucjonizowała NLP i obecnie dominuje w AI. W robotyce stosowana do percepcji wizualnej (Vision Transformers) i przetwarzania multimodalnego.

## Mechanizm Uwagi (Attention)

### Scaled Dot-Product Attention

```python
import numpy as np
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    
    def forward(self, Q, K, V, mask=None):
        """
        Q: Query matrix (batch_size, seq_len, d_k)
        K: Key matrix (batch_size, seq_len, d_k)
        V: Value matrix (batch_size, seq_len, d_v)
        
        Attention(Q,K,V) = softmax(QK^T / √d_k)V
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Apply mask (optional)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
```

### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def split_heads(self, x):
        """
        Split into multiple heads
        (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """
        Combine heads back
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Split into heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Apply attention
        x, attention_weights = self.attention(Q, K, V, mask)
        
        # Combine heads
        x = self.combine_heads(x)
        
        # Final linear projection
        output = self.W_o(x)
        
        return output, attention_weights
```

## Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not trainable)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input
        """
        return x + self.pe[:, :x.size(1)]
```

## Feed-Forward Network

```python
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        FFN(x) = max(0, xW1 + b1)W2 + b2
        """
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

## Transformer Encoder Layer

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x
```

## Complete Transformer Model

```python
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, src, src_mask=None):
        # Embed and add positional encoding
        x = self.src_embedding(src)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Embed and add positional encoding
        x = self.tgt_embedding(tgt)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode
        memory = self.encode(src, src_mask)
        
        # Decode
        output = self.decode(tgt, memory, tgt_mask, src_mask)
        
        # Project to vocabulary
        output = self.fc_out(output)
        
        return output
```

## Vision Transformer (ViT)

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Convolutional patch embedding
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.proj(x)  # (batch_size, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        n_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim,
                num_heads,
                int(embed_dim * mlp_ratio),
                dropout
            )
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoder
        for block in self.blocks:
            x = block(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Classification head (use cls token)
        cls = x[:, 0]
        output = self.head(cls)
        
        return output
```

## Training Example

```python
# Initialize model
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.05
)

# Training loop
model.train()
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## Variacje Transformer

| Model | Zmiana | Zastosowanie |
|-------|--------|--------------|
| **BERT** | Masked LM | NLP pre-training |
| **GPT** | Causal masking | Text generation |
| **ViT** | Image patches | Computer Vision |
| **DETR** | Object queries | Object detection |
| **Perceiver** | Cross-attention | Multimodal |
| **Swin** | Shifted windows | Efficient ViT |

## Powiązane Artykuły

- [LLM](#wiki-llm) - Large Language Models
- [VLM](#wiki-vlm) - Vision-Language Models
- [Deep Learning](#wiki-deep-learning)
- [Neural Networks](#wiki-neural-networks)

## Zasoby

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

---

*Ostatnia aktualizacja: 2025-02-11*  
*Autor: Zespół Kognicji, Laboratorium Robotów Humanoidalnych*
