// Native Kennel Solution 
//  Utilizes 3 Kernal launches to 
//  1. Find Max in each row
//  2. Compute the exponentials of each element and store
//  3. Sum all the exponetials then dvide each by sum

// Kernel 1: Read N (find max)
// Kernel 2: Read N + Write N (compute exp)
// Kernel 3: Read N + Write N (normalize)
// 5N


// Kernal 1: Find max value in each row
__global__ void find_max_kernel(
    const float* x,              // Input matrix 
    float* max_vals,             // Output max values 
    const int batch_size,         // Number of rows
    const int seq_len             // Number of columns per row
){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= batch_size)
        return;
    
    const float* my_row = x + row * seq_len;

    float max_val = -INFINITY;
    for (int i = 0; i < seq_len; i++) {
        max_val = fmaxf(max_val, my_row[i]);
    }
}

// Kernel 2: Compute exponentials for each element
__global__ void exp_kernel(
    const float* x, // Input matrix
    const float* max_vals,
    float * exp_x,
    const int batch_size,
    const int seq_len
){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= batch_size)
        return;
    
    float max_val = max_vals[row];
    const float* my_row = x + row * seq_len;
    float* my_exp = exp_x + row * seq_len;

    for(int i = 0; i < seq_len; i++){
        my_exp[i] = expf(my_row[i] - max_val);
    }
}

// Kernel 3: Sum and normalize 
__global__ void normalize_kernel(
    const float* exp_x,
    float* output,
    const int batch_size,
    const int seq_len
){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= batch_size)
        return;

    const float* my_exp = exp_x + row * seq_len;
    float* my_out = output + row * seq_len;

    // Sum all exponentiated values
    float sum = 0.0f;
    for(int i = 0; i < seq_len; i++){
        sum += my_exp[i];
    }
    // Divide by each sum to normalize
    for(int i = 0; i < seq_len; i++){
        my_out[i] = my_exp[i] / sum;
    }
}
