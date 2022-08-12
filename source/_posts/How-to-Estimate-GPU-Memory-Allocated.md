---
title: How to Estimate GPU Memory Allocated?
date: 2022-08-12 19:20:21
categories: Experimental Skills
---

# How to Estimate GPU Memory Allocated?

This article wants to answer a centeral question in DL experiments: 
Given model architecture, optimizer, batch size, GPU type, how to estimate the GPU memory usage?

use model arch, get number of parameters and activations.
use optimizer and GPU type, 
get number of value need to store for each param, and data type for each value