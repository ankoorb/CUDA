#include <stdio.h>

// Hillis and Steele Scan
__global__ void scan(int *d_hist, const int hist_size){

	// Assumption that only 1 Block with maximum 1024 thread are used
	
	// Mapping from 1D block grid to 1D location on row-major array
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	for (int i = 1; i < hist_size; i *= 2){
		
		// left index position: 1, 2, 4, ..., i.e. 2^i
		int t = idx - i;
		
		// Get the left index position value
		int bin_value = 0;
		if (t >= 0){
			bin_value = d_hist[t];
		}
		
		__syncthreads();
		
		// Add the left index position value to current position value
		if (t >= 0){
			d_hist[idx] += bin_value;
		}
		
		__syncthreads();
	}
	
	// Make this exclusive (above for loop returns inclusive cumulative sum)
	if (idx > 0){
		d_hist[idx] = d_hist[idx - 1];
	}
	else if (idx == 0){
		d_hist[idx] = 0;
	}
	
}



int main(int argc, char **argv){
	// Constants
	const int size = 8;
	const int maxThreadsPerBlock = 8;
	const int maxBlocks = size / maxThreadsPerBlock;
	
	// Create input and output (initialized with 0) arrays
	int h_in[size];
	int h_out[size];
	
	for (int i = 0; i < size; i++){
		h_in[i] = i * 2;
		h_out[i] = 0;
	}
	
	// Print input
    for (int i = 0; i < size; i++){
    	printf("Id %d: Value: %d\n", i, h_in[i]);
    }
	printf("\n");
	
	// Declare device memory pointer
	int *d_cdf;
	
	// Allocate device memory
	cudaMalloc((void **) &d_cdf, sizeof(int) * size);
	
	// Transfer input arrays from host to device
	cudaMemcpy(d_cdf, h_in, sizeof(int) * size, cudaMemcpyHostToDevice);
	
	// Launch kernel
	scan<<<maxBlocks, maxThreadsPerBlock>>>(d_cdf, size);
	
	// Copy the result from device to host
    cudaMemcpy(h_out, d_cdf, sizeof(int) * size, cudaMemcpyDeviceToHost);
	printf("Results copied from device to host.\n");
	
	// Print CDF
    for (int i = 0; i < size; i++){
    	printf("Bin %d: Value: %d\n", i, h_out[i]);
    }
	
}