#include <iostream>

using namespace std;

// Add function to add elements of two arrays
__global__
void add(int n, float *x, float *y){
	for (int i = 0; i < n; i++){
		y[i] = x[i] + y[i];
	}
}

int main(){

	int N = 1<<20; // 1M elements

	float *x;
	float *y;

	// Allocate Unified Memory - Accessible from CPU or GPU
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));

	// Initialize x and y arrays on the host
	for (int i = 0; i < N; i++){
		x[i] = 1.0;
		y[i] = 2.0;
	}

	// Run kernel on 1M elements on the GPU
	add<<<1, 1>>>(N, x, y);

	// Wait for the GPU to finish before accessing on host
	cudaDeviceSynchronize();

	// Free memory
	cudaFree(x);
	cudaFree(y);

	return 0;

}
