#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

// Reduce kernel that uses global memory
__global__ void global_reduce_kernel(float *d_out, float *d_in){
	
	// Mapping from 1D block grid to 1D location on row-major array
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	// Thread ID
	int threadId = threadIdx.x;
	
	// Reduction in global memory
	// NOTE: If s = 1024 then s >>= 1 will return 512, for loop: 512, 256, ..., 2, 1
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
		
		// On each iteration the active region is divided in half: A and B. Now 
		// add A[i] + B[i + len(A)], i.e. Add the value from the corresponding location in other half
		if (threadId < s){
			d_in[idx] += d_in[idx + s];
		}
		
		// Make sure all additions at stage-i are done
		__syncthreads();
	}
	
	// After the loop only 1 element remains. Write that element back to global memory.
	// Only thread 0 writes result for this block to global memory
	if (threadId == 0){
		d_out[blockIdx.x] = d_in[idx];  // Think 1024 blocks used that is why write to blockIdx.x
	}
	
}


// Function to apply reduction stages
void reduce(float *d_out, float *d_intermediate, float *d_in, int size){

	// Assumption: (1) "size" is not greater than maxThreadsPerBlock^2
	// (2) "size" is a multiple of maxThreadsPerBlock
	const int maxThreadsPerBlock = 1024;
	int threads = maxThreadsPerBlock;
	int blocks = size / maxThreadsPerBlock;

	// assert(size < maxThreadsPerBlock * maxThreadsPerBlock);  // Fails for some reason
	assert(size % maxThreadsPerBlock == 0);

	// Launch kernel for stage-i reduction
	global_reduce_kernel<<<blocks, threads>>>(d_intermediate, d_in);

	// Now stage-ii will have one block left to be reduced further
	threads = blocks;  
	blocks = 1;

	// Launch kernel for stage-ii reduction
	global_reduce_kernel<<<blocks, threads>>>(d_out, d_intermediate);
}


int main(int argc, char **argv){

	// Check if GPUs support CUDA
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0){
		fprintf(stderr, "Error: No devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}

	// Print GPU properties
	int dev = 0;
	cudaSetDevice(dev);

	cudaDeviceProp devProps;
	if (cudaGetDeviceProperties(&devProps, dev) == 0){
		printf("Using device %d\n", dev);
		printf("%s; global mem: %ldB; compute v%d.%d; clock: %d Khz\n", 
			   devProps.name, (long int)devProps.totalGlobalMem,
			   (int)devProps.major, (int)devProps.minor,
			   (int)devProps.clockRate);
	}

	// Generate input data on host
	const int ARRAY_SIZE = 1 << 20;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	float h_in[ARRAY_SIZE];
	float sum = 0.0f;

	for (int i = 0; i < ARRAY_SIZE; i++){

		// Generate random float in [-1.0f, 1.0f] and populate the array
		h_in[i] = -1.0f + (float)random() / ((float)RAND_MAX / 2.0f);  // h_in[i] = 0.5f;
		sum += h_in[i];
	}

	// Declare device memory pointers
	float *d_in;
	float *d_intermediate;
	float *d_out;

	// Allocate device memory
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **) &d_intermediate, ARRAY_BYTES);  // Over-allocated
	cudaMalloc((void **) &d_out, sizeof(float));  // Array is reduced to 1 number

	// Transfer array from host to device
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // Launch kernel
    reduce(d_out, d_intermediate, d_in, ARRAY_SIZE);

    // Copy the result from device to host
    float h_out;

    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sequential SUM(h_in): %f\n", sum);
    printf("Parallel SUM: h_out: %f\n", h_out);

    // Deallocate device memory
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);

    return 0;

}

