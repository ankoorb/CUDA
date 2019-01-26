// Compile command: nvcc atomic_rw_same_mem_loc.cu -o atomic_rw_same_mem_loc

#include <stdio.h>
#include "gputimer.h"

#define NUM_THREADS 1000000
#define BLOCK_WIDTH 1000

#define ARRAY_SIZE  100



// Helper function to print an array
void print_array(int *array, int size)
{
    printf("{ ");
    for (int i = 0; i < size; i++) { 
        printf("%d ", array[i]); 
    }
    printf("}\n");
}

// Atomic implementation returns CORRECT result!
__global__ void increment_atomic(int *g)
{
    // Mapping from block grid to thread identity
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    // Each thread to increment consecutive elements, wrapping at ARRAY_SIZE
    i = i % ARRAY_SIZE;  
    atomicAdd(& g[i], 1);
}

int main(int argc,char **argv)
{   
    GpuTimer timer;
    printf("%d total threads in %d blocks writing into %d array elements\n",
           NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE);

    // Declare and allocate host memory
    int h_array[ARRAY_SIZE];
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
 
    // Declare, allocate, and zero out GPU memory
    int * d_array;
    cudaMalloc((void **) &d_array, ARRAY_BYTES);
    cudaMemset((void *) d_array, 0, ARRAY_BYTES); 

    // Launch the kernel 
    timer.Start();
    increment_atomic<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
    timer.Stop();
    
    // Copy the result from device to the host
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // Print resulting array and elapsed time
    print_array(h_array, ARRAY_SIZE);
    printf("Time elapsed = %g ms\n", timer.Elapsed());
 
    // Free GPU memory allocation and exit
    cudaFree(d_array);

    return 0;
}