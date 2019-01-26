#include <stdio.h>
#include <iostream>

#define N 64
#define M 32
#define BLOCK_DIM 32

__global__ void matrixMultiply(int *d_a, int *d_b, int *d_out, int nRows, int nCols){
	
    // Mapping from 2D block grid to absolute 2D locations on C matrix
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  
	// Multiply
	int sum = 0;
	if (idx_x < nRows && idx_y < nRows){
		for (int k=0; k<nRows; k++){
			sum += d_a[nCols * idx_x + k] * d_b[nRows * k + idx_y];
		}
		
	    // 2D location in C matrix to global memory 1D offset
	    int index = idx_y + idx_x * nRows;  // Row-major order with 0 based indices
		d_out[index] = sum;
	}
}

int main(){
	// Declare 2D matrices on host
	int h_a[N][M], h_b[M][N], h_out[N][N];
	
	// Declare device/GPU memory pointers
	int *d_a, *d_b, *d_out;
	
	// Memory size
	int sizeA = N * M * sizeof(int);
	int sizeB = M * N * sizeof(int);
	int sizeC = N * N * sizeof(int);
	
	// Initialize matrices on host
	for (int i=0; i<N; i++){
		for (int j=0; j<M; j++){
			h_a[i][j] = 1;  // Matrix A
		}
	}
	
	for (int i=0; i<M; i++){
		for (int j=0; j<N; j++){
			h_b[i][j] = 2;  // Matrix B
		}
	}
	
	// Allocate GPU memory
	cudaMalloc((void **) &d_a, sizeA);
	cudaMalloc((void **) &d_b, sizeB);
	cudaMalloc((void **) &d_out, sizeC);
	
	// Transfer input matrices from host to device
	cudaMemcpy(d_a, h_a, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeB, cudaMemcpyHostToDevice);
	
	// Define grid blocks dimensions
	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil(N/blockSize.x), (int)ceil(N/blockSize.y));
	
	// Launch the kernel
	matrixMultiply<<<gridSize, blockSize>>>(d_a, d_b, d_out, N, M);
	
	// Copy the result from device to the host
	cudaMemcpy(h_out, d_out, sizeC, cudaMemcpyDeviceToHost);
	
	// Print out the sum of output matrix elements
	int total = 0;
	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++){
			total += h_out[i][j];
		}
	}
	std::cout << "Total: " << total << std::endl;
	
	// Free GPU memory allocation
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);
	
	return 0;
}
