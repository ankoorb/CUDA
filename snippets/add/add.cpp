#include <iostream>

using namespace std;

// Add function
void add(int n, float *x, float *y){
	float value = 0;
	for (int i = 0; i < n; i++){
		y[i] = x[i] + y[i];
		value += y[i];
	}

	cout << "Sum of 2 arrays: " << value << endl;
}

int main(){

	int N = 1<<20; // 1M elements

	float *x = new float[N];
	float *y = new float[N];

	// Initialize x and y arrays on the host
	for (int i = 0; i < N; i++){
		x[i] = 1.0;
		y[i] = 2.0;
	}

	// Run kernel on 1M elements on the CPU
	add(N, x, y);

	// Free memory
	delete [] x;
	delete [] y;

	return 0;

}
