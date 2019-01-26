#include <stdio.h>
#include <iostream>

#define N 64
#define BLOCK_DIM 32

__global__ void matrixTranspose(int *d_a, int *d_out, int width){
	
    // Mapping from 2D block grid to absolute 2D locations on C matrix
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
	// Transpose
	if (row < width && col < width){		
		d_out[row * width + col] = d_a[col * width + row];  
	}
}

int main(){
	// Declare 2D matrices on host
	int h_a[N][N], h_out[N][N];
	
	// Declare device/GPU memory pointers
	int *d_a, *d_out;
	
	// Memory size
	int size = N * N * sizeof(int);
	
	// Initialize matrices on host
	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++){
			if (i % 2 == 0){
				h_a[i][j] = i + j;  // Matrix A
			}
			else {
				h_a[i][j] = i * j;  // Matrix A
			}
			
		}
	}
	
	// Allocate GPU memory
	cudaMalloc((void **) &d_a, size);
	cudaMalloc((void **) &d_out, size);
	
	// Transfer input matrices from host to device
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	
	// Define grid blocks dimensions
	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil(N/blockSize.x), (int)ceil(N/blockSize.y));
	
	// Launch the kernel
	matrixTranspose<<<gridSize, blockSize>>>(d_a, d_out, N);
	
	// Copy the result from device to the host
	cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

	// Print out the transposed array
	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++){
			printf("A[%d,%d]: %d\t", i, j, h_a[i][j]);
			printf("AT[%d,%d]: %d", i, j, h_out[i][j]);
			printf("\n");
		}
	}
	
	// Free GPU memory allocation
	cudaFree(d_a);
	cudaFree(d_out);
	
	return 0;
}
