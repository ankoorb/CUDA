#include <stdio.h>

#define ARRAY_SIZE 128

// Helper function to print an array
void print_array(float *array, int array_size) {
    printf("{ ");
    for (int i = 0; i < array_size; i++) { 
        printf("%0.2f, ", array[i]); 
    }
    printf("}\n");
}

// Using shared memory (For clarity, hardcoding 128 threads/elements and omitting out-of-bounds checks)
__global__ void use_shared_memory(float *array)
{
    // Local variables (private to each thread)
    int index = threadIdx.x;
    float sum = 0.0f;
    float average; 

    // __shared__ variables are visible to all threads in the thread block
    // and have the same lifetime as the thread block
    __shared__ float sh_arr[ARRAY_SIZE];

    // Copy data from "array" in global memory to sh_arr in shared memory.
    // Here, each thread is responsible for copying a single element.
    sh_arr[index] = array[index];

    __syncthreads();    // Ensure all the writes to shared memory have completed

    // Now, sh_arr is fully populated. Let's find the average of all previous elements
    for (int i=0; i<index; i++) { 
        sum += sh_arr[i]; 
    }

    // Average of array[0..index-1]
    if (index > 0) {
        average = sum / (index + 0.0f);
    } else {
        average = 0.0f;
    }

    // If array[index] is greater than the average of array[0..index-1], replace with average.
    // since array[] is in global memory, this change will be seen by the host (and potentially 
    // other thread blocks, if any)
    if (array[index] > average) {  
        array[index] = average; 
    }

    //printf("Thread: %d, and average: %0.2f\n", index, average);

    // The following code has NO EFFECT: it modifies shared memory, but 
    // the resulting modified data is never copied back to global memory
    // and vanishes when the thread block completes
    sh_arr[index] = 3.14;
}

int main(int argc, char **argv)
{

    // Declare 1D array on host
    float h_arr[ARRAY_SIZE];  

    // Declare device memory pointer 
    float *d_arr;   

    // Memory size
    int SIZE = ARRAY_SIZE * sizeof(float);

    // Initialize 1D array on host
    for (int i=0; i<ARRAY_SIZE; i++) {
        if (i % 2) { h_arr[i] = i * 2.5f; } else { h_arr[i] = 0.0f; }
    }  

    // Print the input array
    print_array(h_arr, ARRAY_SIZE); 

    // Allocate device memory
    cudaMalloc((void **) &d_arr, SIZE);

    // Transfer input array from host memory to device memory
    cudaMemcpy((void *)d_arr, (void *)h_arr, SIZE, cudaMemcpyHostToDevice);

    // Launch kernel
    use_shared_memory<<<1, ARRAY_SIZE>>>(d_arr); 

    // Copy the result from device to the host
    cudaMemcpy((void *)h_arr, (void *)d_arr, SIZE, cudaMemcpyDeviceToHost);

    // Print the output array
    print_array(h_arr, ARRAY_SIZE);

    // Free GPU memory allocation
    cudaFree(d_arr);

    return 0;
}


















