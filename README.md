## Overview

This website accompanies our TMLR survey, **“A Survey of Flow Matching in Reinforcement Learning.”** It is intended as a starting point for researchers and newcomers interested in the rapidly growing intersection of flow matching and reinforcement learning.

We organize key papers, methods, code repositories, and learning resources to make the FM-RL literature easier to understand and navigate.

## Paper

### A Survey of Flow Matching in Reinforcement Learning

**Nabuat Zaman Nahim, Fairoz Nower Khan, Peizhong Ju**  
*Transactions on Machine Learning Research (TMLR), 2026* - **Survey Certification**

[![arXiv](https://img.shields.io/badge/arXiv-2610.16275-b31b1b.svg)](https://arxiv.org/abs/2610.16275)
[![TMLR](https://img.shields.io/badge/TMLR-Published-2ea44f.svg)](https://tmlr.infinite-conf.org/paper_pages/P6e5IC4gPe.html)
[![OpenReview](https://img.shields.io/badge/OpenReview-Reviews-8c1b13.svg)](https://openreview.net/forum?id=P6e5IC4gPe)
[![PDF](https://img.shields.io/badge/PDF-Paper-blue.svg)](https://arxiv.org/pdf/2610.16275)
[![BibTeX](https://img.shields.io/badge/BibTeX-Citation-orange.svg)](https://www.jmlr.org/tmlr/papers/bib/P6e5IC4gPe.bib)

## Getting Started

If you are new to Flow Matching in Reinforcement Learning, we suggest exploring the material in the following order:

1. **Flow Matching Basics**: Understand the core idea of learning a velocity field that transports samples from a simple source distribution to a target distribution.
2. **Why Flow Matching for RL?**: See why expressive flow based policies can be useful for multimodal action distributions and efficient generation.
3. **FM-RL Taxonomy**: Learn how existing methods differ in what they model and how reinforcement learning signals are incorporated.
4. **Key Papers**: Explore a curated collection of representative FM-RL methods, organized by category.
5. **Code & Resources**: Find available implementations, project pages, and additional learning materials.

## Flow Matching Basics

**Flow Matching (FM)** is a generative modeling framework for learning how to transform samples from a simple distribution, such as Gaussian noise, into samples from a complex target distribution.

The main idea is to learn a **time-dependent velocity field** that tells each sample how it should move over time. Starting from an initial sample \(x_0\), generation follows an ordinary differential equation (ODE):

$$
\frac{dx_t}{dt} = v_\theta(x_t,t)
$$

where $v_\theta(x_t,t)$ is a neural network that predicts the velocity of the sample $x_t$ at time $t$.

Intuitively, you can think of Flow Matching as learning a **transport map**:

**simple noise distribution → learned continuous flow → target data distribution**

<p align="center">
  <img src="assets/figures/flow-matching-nutshell.svg"
       alt="Flow Matching in a nutshell"
       width="90%">
</p>

<p align="center">
  <em>Figure 1: Flow Matching in a nutshell. Training is simulation-free, while generation is performed by integrating the learned ODE from the base distribution to the target distribution.</em>
</p>

Unlike diffusion models, which are commonly described through iterative stochastic noising and denoising processes, Flow Matching directly learns the vector field governing this transport. This provides a convenient way to build expressive generative models while using an ODE for sampling.
