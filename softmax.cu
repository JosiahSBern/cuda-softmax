#include <cuda_runtime.h>
#include <math.h>


__global__ void softmax_kernel(
    const float* input,
    float* output,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    // Shared memory for max and sum
    extern __shared__ float shared[];
    float* shared_max = shared;
    float* shared_sum = shared + blockDim.x;

    shared_max[tx] = -INFINITY;
    shared_sum[tx] = 0.0f;

    if(i < N){
        shared_max[tx] = input[i];
    }
    __syncthreads();


    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if(tx < stride) {
            shared_max[tx] = fmaxf(shared_max[tx], shared_max[tx + stride]);
        }
    }
    __syncthreads();

    float max_val = shared_max[0];

    __syncthreads();
    shared_sum[tx] = 0.0f;

    if(i < N){
        float exp_val = expf(input[i] - max_val);
        shared_sum[tx] = exp_val;
    }
    __syncthreads();


    for (int stride = blockDim.x / 2; stride > 0; stride /= 2 ){
        if(tx < stride){
            shared_sum[tx] = shared_sum[tx] + shared_sum[tx + stride];
        }
        __syncthreads();
    }
    float sum_val = shared_sum[0];

    // Normalize
    if(i < N){
        output[i] = expf(input[i] - max_val) / sum_val;
    }

}

cudaError_t softmax_32(
    const float* input,
    float* output,
    int N
){
    int block_size = 256;
    int grid_size = (N + block_size - 1 ) / block_size;

    size_t shared_mem = 2 * block_size * sizeof(float);
    softmax_kernel <<< grid_size,block_size,shared_mem>>>(
        input, output, N
    );

    return cudaGetLastError();
}


__global__ void softmax_batched_kernel(
    const float* input,
    float* output,
    int batch,
    int N
){
    int batch_idx = blockIdx.x;
    int tx = threadIdx.x;

    if(batch_idx >= batch)
        return;
    
    const float* my_input = input + batch_idx * N;
    float* my_output = output + batch_idx * N;

    extern __shared__ float shared[];
    float* shared_max = shared;
    float* shared_sum = shared + blockDim.x;

    shared_max[tx] = - INFINITY;
    shared_sum[tx] = 0.0f;

    if (tx < N) {                          
        shared_max[tx] = my_input[tx];     
    }
    __syncthreads();
    
    // Parallel reduction 
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tx < stride && tx + stride < N) {
            shared_max[tx] = fmaxf(shared_max[tx], shared_max[tx + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_max[0];
    
    __syncthreads();
    shared_sum[tx] = 0.0f;
    
    if (tx < N) {
        shared_sum[tx] = expf(my_input[tx] - max_val);
    }
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            shared_sum[tx] = shared_sum[tx] + shared_sum[tx + stride];
        }
        __syncthreads();
    }
    float sum_val = shared_sum[0];
    
    // Normalize
    if (tx < N) {
        my_output[tx] = expf(my_input[tx] - max_val) / sum_val;
    }
}

    cudaError_t softmax_batched_fp32(
        const float* d_input,
        float* d_output,
        int batch,
        int N
    ) {
        int block_size = 256;
        size_t shared_mem = 2 * block_size * sizeof(float);
        
        // Launch 'batch' blocks (one per vector)
        // Each block handles one entire softmax problem
        softmax_batched_kernel<<<batch, block_size, shared_mem>>>(
            d_input, d_output, batch, N
        );
        
        return cudaGetLastError();
}

