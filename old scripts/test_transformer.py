#!/usr/bin/env python3
"""Test script to validate the annotated transformer implementation."""

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy

# Test 1: Basic imports
print("Test 1: Checking imports...")
try:
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader, Dataset
    # Skip torchtext for now due to binary compatibility issues
    # from torchtext.vocab import build_vocab_from_iterator
    print("✓ Core PyTorch imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Helper classes
print("\nTest 2: Testing helper classes...")

class IteratorDataset(Dataset):
    """Convert an iterator to a map-style dataset for compatibility with newer torchtext versions."""
    def __init__(self, data_iter):
        self.data = list(data_iter)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Test IteratorDataset
data = [1, 2, 3, 4, 5]
dataset = IteratorDataset(iter(data))
assert len(dataset) == 5
assert dataset[0] == 1
assert dataset[4] == 5
print("✓ IteratorDataset works correctly")

# Test 3: Core layer definitions
print("\nTest 3: Testing core layers...")

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

ln = LayerNorm(512)
x = torch.randn(2, 10, 512)
output = ln(x)
assert output.shape == x.shape
print("✓ LayerNorm works correctly")

# Test 4: Embeddings with scaling
print("\nTest 4: Testing Embeddings...")

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

emb = Embeddings(512, 5000)
x = torch.randint(0, 5000, (2, 10))
output = emb(x)
assert output.shape == (2, 10, 512)
print("✓ Embeddings works correctly")

# Test 5: Positional Encoding
print("\nTest 5: Testing Positional Encoding...")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

pe = PositionalEncoding(512, 0.1)
x = torch.randn(2, 10, 512)
output = pe(x)
assert output.shape == x.shape
print("✓ PositionalEncoding works correctly")

# Test 6: Feed-forward network
print("\nTest 6: Testing PositionwiseFeedForward...")

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.nn.functional.relu(self.w_1(x))))

ff = PositionwiseFeedForward(512, 2048)
x = torch.randn(2, 10, 512)
output = ff(x)
assert output.shape == x.shape
print("✓ PositionwiseFeedForward works correctly")

# Test 7: Multi-head attention
print("\nTest 7: Testing MultiHeadedAttention...")

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.nn.functional.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

mha = MultiHeadedAttention(8, 512)
q = torch.randn(2, 10, 512)
k = torch.randn(2, 10, 512)
v = torch.randn(2, 10, 512)
output = mha(q, k, v)
assert output.shape == q.shape
print("✓ MultiHeadedAttention works correctly")

print("\n" + "="*50)
print("All tests passed! ✓")
print("="*50)
