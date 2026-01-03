# The Annotated Transformer - Marimo Notebook

A complete line-by-line implementation of the "Attention is All You Need" paper (Vaswani et al., 2017), rebuilt as an interactive marimo notebook.

## Overview

This notebook contains a fully functional Transformer implementation with:

- **Complete Model Architecture**
  - Encoder with 6 stacked layers
  - Decoder with 6 stacked layers
  - Multi-head attention mechanism (8 heads)
  - Position-wise feed-forward networks
  - Positional encoding with sinusoidal functions
  - Learnable embeddings

- **Training Components**
  - Batch processing with masking
  - Label smoothing regularization
  - Custom learning rate schedule (warmup + inverse sqrt decay)
  - Support for gradient accumulation
  - Training loop with validation

- **Example: Copy Task**
  - Demonstrates complete training pipeline
  - 2-layer transformer trained on synthetic data
  - Tests greedy decoding for inference

## Installation

1. Install dependencies:
```bash
pip install marimo torch pandas numpy
```

2. Run the notebook:
```bash
marimo run annotated_transformer.py
```

Or for interactive development:
```bash
marimo edit annotated_transformer.py
```

## Usage

The notebook is organized into main sections:

### Part 1: Model Architecture
- Core components: Encoder, Decoder, Attention, Feed-Forward
- Building blocks for constructing a complete transformer
- The `make_model()` function creates a full model from hyperparameters

### Part 2: Model Training
- Batch preparation with padding masks and causal masks
- Label smoothing implementation
- Learning rate scheduler from the paper
- Training loop with loss computation

### Part 3: Practical Example
- Synthetic copy task dataset
- Full training pipeline (10 epochs)
- Inference with greedy decoding
- Example output validation

## Key Features

**Complete Implementation**: Every component from the paper is included:
- Scaled dot-product attention
- Multi-head attention with proper projections
- Position-wise feed-forward networks
- Positional encoding
- Layer normalization
- Residual connections

**Proper Masking**: 
- Padding masks to ignore `<blank>` tokens
- Causal masks to prevent attending to future positions
- Proper mask broadcasting across attention heads

**Training Best Practices**:
- Xavier uniform initialization for all weights
- Learning rate warmup schedule
- Label smoothing with confidence-based targets
- Efficient attention computation

## Extending the Implementation

To build on this foundation:

1. **Real Datasets**: Add data loaders for WMT or Multi30k
2. **Beam Search**: Implement k-best decoding
3. **Attention Visualization**: Visualize which positions attend to each other
4. **Distributed Training**: Add multi-GPU support with DDP
5. **Checkpointing**: Save and load model states
6. **BLEU Evaluation**: Add automatic evaluation metrics

## Hyperparameters

Default configuration (as in the paper):
- Model dimension: 512
- FFN inner dimension: 2048
- Number of heads: 8
- Number of layers: 6
- Dropout: 0.1
- Label smoothing: 0.1

## References

- **Paper**: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- **Original Blog**: [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- **Harvard NLP**: [nlp.seas.harvard.edu](https://nlp.seas.harvard.edu/)

## Credits

- Original implementation: Sasha Rush
- Updates and improvements (v2022): Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, Stella Biderman
- Marimo notebook adaptation: 2025

## License

This is an educational implementation based on the annotated-transformer project. Use freely for learning and research purposes.
