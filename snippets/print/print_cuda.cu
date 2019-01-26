#include <stdio.h>

// Function to print from device
__global__
void print(){
  printf("Block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main(){

	// Run kernel on the GPU
	print<<<2, 16>>>();

	return 0;
}
