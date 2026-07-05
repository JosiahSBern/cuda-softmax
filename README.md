# CUDA Softmax
A GPU-optimized softmax kernel implementation.
---
## Softmax Algorthim:
### Understanding Softmax
Softmax converts any list of numbers (raw scores) into probabilities. Probabilities must sum up to 1 and be positive.

Softmax uses an exponential trick to normalize these values. Instead of picking the biggest number, it gives higher weight to bigger raw scores into a probability distributions.

**The basic formula:**
$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}$$

**The problem:** If $x_i$ is large (like 1000), then $e^{x_i}$ overflows and crashes.

**The numerically stable formula:**
$$\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^n e^{x_j - \max(x)}}$$

By subtracting max(x), we keep the exponentials small and avoid overflow.

### The GPU Problem

Softmax requires:
1. Find max (scan the whole vector)
2. Subtract max from everything
3. Compute exp
4. Sum all the exps
5. Divide everything by the sum

We optimize this using parallel reduction. Instead of one thread reading the whole vector, use 256 threads simultaneously with `__syncthreads()` to share results.

This turns 5 passes into 2 passes:
- **Pass 1:** Find max using parallel reduction tree
- **Pass 2:** Compute exp and sum using parallel reduction tree
---
## References
https://www.bvisser.me/blog/softmax-kernel
https://maharshi.bearblog.dev/optimizing-softmax-cuda/
https://medium.com/@dcbaslani/beating-pytorch-writing-a-faster-softmax-kernel-in-cuda-0d0a237cda57
https://www.youtube.com/watch?v=oJU6-qW6xZU
