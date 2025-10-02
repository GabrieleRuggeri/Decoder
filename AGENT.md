# AGENT.md

## Purpose
This document provides guidelines for a coding agent to assist in the development and refinement of a decoder-only transformer implementation (GPT-like) using PyTorch. The goal is to ensure efficient collaboration and adherence to best practices in deep learning and PyTorch development.

---

## Project Overview
The project involves implementing a decoder-only transformer model inspired by GPT architecture. The implementation includes components such as masked multi-head attention, feed-forward layers, residual connections, layer normalization, and token/positional embeddings. The model will be trained on text data to perform language modeling tasks.

---

## Agent Responsibilities

### 1. Code Assistance
- **Component Implementation**: Assist in creating PyTorch modules for various components, such as:
  - Multi-Headed Masked Attention
  - Feed-Forward Networks
  - Residual Layers
  - Layer Normalization
  - Token and Positional Embeddings
- **Model Assembly**: Help integrate components into a cohesive decoder-only transformer architecture.

### 2. Debugging and Optimization
- **Debugging**: Identify and fix issues in the implementation, such as shape mismatches, gradient flow problems, or incorrect masking.
- **Performance Optimization**: Suggest improvements for training efficiency, such as mixed precision training, gradient checkpointing, or optimized data loading.

### 3. Training Support
- **Data Preparation**: Assist in tokenizing and batching text data for training.
- **Training Loop**: Help refine the training loop, including loss computation, backpropagation, and optimizer updates.
- **Monitoring**: Provide tools or scripts to monitor training metrics like loss and perplexity.

### 4. Inference and Evaluation
- **Inference**: Implement functions to generate text given a prompt.
- **Evaluation**: Suggest methods to evaluate the model, such as perplexity computation or BLEU score for generated text.

### 5. Documentation and Best Practices
- **Code Documentation**: Ensure all functions and classes are well-documented with docstrings.
- **Best Practices**: Adhere to PyTorch and deep learning best practices, such as proper weight initialization, gradient clipping, and checkpointing.

---

## Guidelines for Collaboration

### Communication
- Clearly explain the purpose of each request or question.
- Provide relevant context, such as code snippets, error messages, or expected behavior.

### Code Standards
- Follow Python coding conventions (PEP 8).
- Use type hints for function signatures.
- Write modular and reusable code.

### Tools and Libraries
- **Primary Framework**: PyTorch
- **Additional Libraries**: NumPy, Tokenizers (e.g., Hugging Face), and any other relevant tools for text processing and evaluation.

### File Structure
- Maintain a clear and organized file structure. Suggested structure:
  ```
  ├── model/
  │   ├── layers.py       # Individual model components
  │   ├── transformer.py  # Full transformer model
  ├── data/
  │   ├── tokenizer.py    # Tokenization utilities
  │   ├── dataset.py      # Dataset and DataLoader
  ├── train.py            # Training script
  ├── evaluate.py         # Evaluation script
  ├── utils.py            # Utility functions
  ```

---

## Example Tasks

### Task 1: Implement a Masked Multi-Head Attention Layer
- Ensure proper masking to prevent information leakage.
- Validate the implementation with unit tests.

### Task 2: Optimize Training Loop
- Profile the training loop to identify bottlenecks.
- Implement gradient accumulation for large batch sizes.

### Task 3: Add Inference Functionality
- Create a function to generate text given a prompt.
- Ensure the function handles variable-length prompts and batch sizes.

---

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## Notes
- Always validate changes with appropriate tests.
- Prioritize readability and maintainability in the codebase.
- Keep the user informed about progress and any challenges encountered.