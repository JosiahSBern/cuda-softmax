#include <cuda_runtime.h>
#include <math.h>


__global__ void softmax_kernel(
    const float* input,
    float* output,
    int N
) {
    int i = blockIdx.x * 
}

// Compute softmax for batched input

