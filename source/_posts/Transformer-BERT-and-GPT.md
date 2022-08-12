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
The main idea is constructing $t_i$ 's repr based on $\{t_0,...,t_{i-1}\}$. 

Adding \<eos\> in front to construct seq and appending \<bos\> at tail to construct label(essentially a shifting trick), 
also equiqqed with CausalAttention(essentially a triu mask),
we can self-supvervise a whole text in a single path in teacher's force fashion(and multiple texts in a batch fashion).
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
