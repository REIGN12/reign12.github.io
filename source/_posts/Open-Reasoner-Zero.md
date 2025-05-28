---
title: Open Reasoner Zero
date: 2025-05-24 20:18:25
math: true
tags:
---

# Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model

## Introduction
Large-scale reinforcement learning (RL) training of language models on reasoning tasks has emerged as a promising paradigm for mastering complex problem-solving skills.  
In this blog, we introduce Open-Reasoner-Zero, the first open source implementation of large-scale reasoning-oriented RL training on the base model focusing on scalability, simplicity and accessibility.  
Using the same base model, Qwen2.5-32B base, as DeepSeek-R1-Zero-Qwen-32B, our implementation achieves superior performance across AIME2024, MATH500, and GPQA Diamond, while demonstrating remarkable efficiency—requiring only 1/10 of the training steps.  
We release our source code, training data, and various model weights, fostering reproducibility and encouraging further exploration of the properties of related models.

![Benchmark performance of Open-Reasoner-Zero-\{7B, 32B\} during training. Using the same base model, Qwen2.5-32B base, as DeepSeek-R1-Zero-Qwen-32B, Open-Reasoner-Zero-32B achieves superior performance on AIME2024, MATH500, and GPQA Diamond benchmarks—requiring only a tenth of the training steps.](./Open-Reasoner-Zero/1_orz_teaser_0321.png)


<!-- This is the way to actually render the image in the hexo blog -->
<!-- <figure>
  <img src="{% asset_path 1_orz_teaser_0321.png %}"">
  <figcaption>Evaluation performance of Open-Reasoner-Zero-\{7B, 32B\} on benchmarks (averaged on 16 responses) during training. Using the same base model, Qwen2.5-32B base, as DeepSeek-R1-Zero-Qwen-32B, Open-Reasoner-Zero-32B achieves superior performance on AIME2024, MATH500, and GPQA Diamond benchmarks—requiring only a tenth of the training steps.</figcaption>
</figure> -->


## Scale-up Reinforcement Learning on a Base Model
In this section, we describe the strategy and critical components for scale-up reasoning-oriented RL directly on a base model.  
Concretely, we show that a minimalist approach, vanilla PPO with GAE ($\lambda=1$, $\gamma=1$) and straightforward rule-based rewards, without any KL regularization, is sufficient to scale up both benchmark performance and response length.   
Moreover, we detail the fundamental yet critical implementation settings for our approach, covering data curation, prompt design, and reward function specification.

### Choosing PPO over GRPO
We select PPO over GRPO due to its superior value estimation enabled by a learned critic. This critic facilitates accurate token-level value estimation, effectively identifying and devaluing detrimental patterns such as repetitive behaviors, named credit assignment. Consequently, PPO achieves notably more robust advantage estimation compared to GRPO.   
This deficiency can misdirect reinforcement, leading to training instability and eventual collapse, an observation supported by community discussions (OpenR1: discussion about vanilla GRPO reproduction [link](https://huggingface.co/spaces/open-r1/README/discussions/20\#67ef94b84e6c9e7404c1e1df)). Detailed analysis is also provided in Experiments section.

### Algorithm Implementations
Our empirical studies suggests that vanilla PPO already provides a highly stable and robust training across different model scales and training durations.  
Nonetheless, appropriate implementations matter. 
Through extensive experiments, we found that the choice of GAE parameters substantially impacts performance in reasoning-oriented tasks. 
Specifically, the discount factor $\gamma$ controls the effective sequence length considered during training: a lower $\gamma$ assigns exponentially decreasing weights to future rewards, inducing the model to prematurely terminate generation in order to more immediately obtain rewards.
On the other hand, the GAE parameter $\lambda$ balances bias and variance in advantage estimation. Crucially, in large-scale training scenarios, the substantial data volume naturally mitigates variance concerns, encouraging us to adopt a bias-free configuration.
Consequently, by setting $\gamma=1$ and $\lambda=1$, we fully capture the long-term dependencies critical for reasoning tasks and achieve stable training.
Fortuitously, this also leads to a significant simplification of the GAE advantage computation in our case:   
$$ 
\begin{align}
\hat{A}\_t^{GAE(\gamma=1, \lambda=1)} = R - V\_\phi(s_t), \\\\
\mathcal{J}\_{\text{value}}(\phi) = \frac{1}{2}\mathbb{E}\_{\tau \sim \pi\_{\theta\_{\text{old}}}} \left[ \sum\_{t=0}^{T-1} (V\_\phi(s\_t) - R)^2 \right], 
\end{align}
$$
where $R$ is the single terminal reward. 


### Removing KL regularization
We achieve stable training without relying on any KL-based regularization techniques (\eg, KL shaped rewards and loss), different from the de facto RLHF community and Reasoner model. 
Intuitively, KL regularization constrains the policy model to remain close to the original base model distribution, potentially limiting exploration during policy optimization. 
By omitting KL regularization, our approach offers several practical advantages: (1) it obviates the need to navigate the large and challenging-to-tune design space inherent to KL regularization, greatly simplifying the training procedure; and (2) it lowers computational overhead and memory usage, eliminating the need to load the weight of a separate reference model and calculate log probabilities using it. 
Together, these benefits facilitate efficient and scalable large-scale RL training.

### Minimal Reward Function Design
In contrast to approaches such as DeepSeek R1, which utilize a dedicated format reward to enforce structured reasoning (e.g., enclosing thought processes within <think>...</think>), we demonstrate that the simplest, rule-based reward function is not only sufficient but also optimal, as minimal design leaves no room for potential reward hacking.
Notably, even unaligned base models quickly adpot to desired format, suggesting this is a straightforward task without requiring complex reward engineering.

### Scale up Training Data.
We identify that scaling up data quantity and diversity is pivotal for Reasoner-Zero training. While training on limited academic datasets like MATH train set leads to quick performance plateaus, our curated large-scale diverse dataset demonstrates impressive potential for continuous improvement without signs of saturation on both training and test sets.



## Use ORZ Models
```python
# TODO
```

## Future Work
