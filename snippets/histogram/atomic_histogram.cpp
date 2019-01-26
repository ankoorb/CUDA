#include <stdio.h>
#include <cuda_runtime.h>

// Helper functions to generate data
int log2(int i){
	int r = 0;
	while (i >>= 1){
		r++;
	}
	return r;
}

int bit_reverse(int w, int bits){
	int r = 0;
	for (int i = 0; i < bits; i++){
		int bit = (w & (1 << i)) >> i;
		r |= bit << (bits - i - 1);
	}
	return r;
}

// Histogram (uses atomicAdd)
__global__ void atomic_histogram(int *d_bins, const int *d_in, const int BIN_COUNT){
	
	// Mapping from 1D block grid to 1D location on row-major array
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	// Get item from input array
	int item = d_in[idx];

	// Compute bin id to update its count
	int bin_id = item % BIN_COUNT;

	// Update count using atomicAdd
	atomicAdd(&(d_bins[bin_id]), 1);
	
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

	// Constants
	const int maxThreadsPerBlock = 1024;
	const int ARRAY_SIZE = 65535;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
	const int BIN_COUNT = 16;
	const int BIN_BYTES = BIN_COUNT * sizeof(int);

	// Generate input arrays (data) on host
	int h_in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++){
		h_in[i] = bit_reverse(i, log2(ARRAY_SIZE));
	}

	int h_bins[BIN_COUNT];
	for (int i = 0; i < BIN_COUNT; i++){
		h_bins[i] = 0; // Initialize bins with 0
	}

	// Declare device memory pointers
	int *d_in;
	int *d_bins;

	// Allocate device memory
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **) &d_bins, BIN_BYTES);

	// Transfer input arrays from host to device
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bins, h_bins, BIN_BYTES, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = maxThreadsPerBlock;
    int blocks = ARRAY_SIZE / maxThreadsPerBlock;
    atomic_histogram<<<threads, blocks>>>(d_bins, d_in, BIN_COUNT);

    // Copy the result from device to host
    cudaMemcpy(h_bins, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);

    // Print histogram
    for (int i = 0; i < BIN_COUNT; i++){
    	printf("Bin %d: Count: %d\n", i, h_bins[i]);
    }

    // Deallocate device memory
    cudaFree(d_in);
    cudaFree(d_bins);

    return 0;

}

