# Mini_GPT: A Transformer-Based Language Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
![Transformers](https://img.shields.io/badge/Architecture-Transformer-yellowgreen)

An end-to-end implementation of a GPT-style language model built from scratch during summer internship under NEP 2020 guidelines.

## 📌 Project Overview
- **Type**: Graded On-Job Training Project
- **Duration**: Summer Break 2023
- **Mentorship**: Industry Professionals
- **Key Feature**: No APIs used - every transformer component implemented manually

## 🛠️ Technical Implementation
```python
# Example code structure
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        ...
