---
title: Transformer, BERT, and GPT
date: 2022-08-12 20:22:00
categories: Basic Concepts
---

# Gentle Introduction to Transformer, BERT and GPT

## GPT

minimal example:
```python
<bos>, t -> t, <eos>
a12 = 0
```
```python
class Dataset:
    def __getitem__(self,text:List):
        seq = [<bos>] + text
        label = text + [<eos>]
        return seq, label

class CausalAttention(nn.Module):
    def forward(self,x:Tensor):
        B,TL,D = x.shape
        mask = torch.tril(torch.ones(TL,TL)) # rm a_{ij}, j>i 's connection





```
