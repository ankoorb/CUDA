/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <stdio.h>      /* printf */
#include <math.h>       /* ceil */

// Reduce Min or Max Kernel (shared memory)
__global__ void reduce_min_max(float *d_out, const float* const d_in, bool compute_min){

  // Shared memory is allocated in the kernel call (3rd argument): <<<blocks, threads, shared_memory>>>
  extern __shared__ float s_data[];

  // Mapping from 1D block grid to 1D location on row-major array
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  
  // Thread ID
  int threadId = threadIdx.x;

  // Load shared memory from global memory. NOTE: threads share shared memory
  s_data[threadId] = d_in[idx];

  // Make sure entire block is loaded (i.e. all read/write operations are done)
  __syncthreads();

  /* for (int s = 8 / 2; s > 0; s >>= 1){
        cout << s << endl;
    }
  >>> 4
  >>> 2
  >>> 1
  NOTE: If s = 1024 then s >>= 1 will return 512, for loop: 512, 256, ..., 2, 1
  */
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
    
    // On each iteration the active region is divided in half: A and B. Now compute
    // min or max of (A[i], B[i + len(A)]), i.e. Compare the value with the corresponding location in other half
    if (threadId < s){

      if (compute_min){

        s_data[threadId] = min(s_data[threadId], s_data[threadId + s]);  // Reading and writing to shared memory

      } else {

        s_data[threadId] = max(s_data[threadId], s_data[threadId + s]);  // Reading and writing to shared memory

      }
      
    }
    
    // Make sure all additions at stage-i are done
    __syncthreads();
  }

  // After the loop only 1 element remains. Write that element back to global memory.
  // Only thread 0 writes result for this block to global memory
  if (threadId == 0){

    d_out[blockIdx.x] = s_data[threadId];  // Think 1024 blocks used that is why write to blockIdx.x

  }

}

// Histogram (atomicAdd based) 
__global__ void histogram_atomic(unsigned int *d_out, 
                                 const float* const d_in, 
                                 const size_t n_bins, 
                                 const float min_log_lum, 
                                 const float range_log_lum)
{

  // Mapping from 1D block grid to 1D location on row-major array
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // Compute bin id to update its count, Formula: bin = (lum[i] - lumMin) / lumRange * numBins
  int bin_id = (d_in[idx] - min_log_lum) / range_log_lum * n_bins;

  // What happens when bin_id == n_bins? Since C++ uses Zero based numbering, there is no bin at n_bins th index
  if (bin_id == n_bins){
    bin_id--;
  }

  // Update count using atomicAdd
  atomicAdd(&(d_out[bin_id]), 1);

}

// Scan Hillis/Steele Scan
// Shifting array data (Page-10: http://users.umiacs.umd.edu/~ramani/cmsc828e_gpusci/ScanTalk.pdf)
__global__ void hillis_steele_scan(unsigned int *d_in, const size_t n_bins){

  // Assumption that only 1 Block with maximum 1024 thread are used

  // Mapping from 1D block grid to 1D location on row-major array
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // Exclusive scan: shift right by one and set first element to 0
  if (idx > 0){
    d_in[idx] = d_in[idx-1];
  } else {
    d_in[idx] = 0; // First element set to 0
  }
  __syncthreads();

  // Scan loop is inclusive, but shifting item to right returns exclusive output
  for (int i = 1; i < n_bins; i *= 2){

    // left index position: 1, 2, 4, ..., i.e. 2^i
    int t = idx - i;

    // Get the left index position value
    int bin_value = 0;
    if (t >= 0){
      bin_value = d_in[t];
    }
    __syncthreads();

    // Add the left index position value to current position value
    if (t >= 0){
      d_in[idx] += bin_value;
    }
    __syncthreads();
  }

}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
  */

  // Constants
  const int maxThreadsPerBlock = 1024;
  const int threads = maxThreadsPerBlock;
  const int blocks = ceil((float)(numRows * numCols)/threads);
  const int intermediate_bytes = sizeof(float) * blocks;

  // Declare device memory pointers for min and max computation
  float *d_intermediate;
  float *d_min;
  float *d_max;

  // Allocate device memory
  cudaMalloc((void **) &d_intermediate, intermediate_bytes);
  cudaMalloc((void **) &d_min, sizeof(float));
  cudaMalloc((void **) &d_max, sizeof(float));

  // Min computation
  // ---------------
  // Launch kernel for stage-i reduction: Shared memory size: Number of threads * data type
  reduce_min_max<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_logLuminance, true);

  // Now stage-ii will have one block left to be reduced further
  reduce_min_max<<<1, blocks, blocks * sizeof(float)>>>(d_min, d_intermediate, true);

  // Max computation
  // ---------------
  // Launch kernel for stage-i reduction: Shared memory size: Number of threads * data type
  reduce_min_max<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_logLuminance, false);

  // Now stage-ii will have one block left to be reduced further
  reduce_min_max<<<1, blocks, blocks * sizeof(float)>>>(d_max, d_intermediate, false);

  // Wait for the GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Copy the result from device to host
  cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost);

  printf("min_logLum: %f\n", min_logLum);
  printf("max_logLum: %f\n", max_logLum);

  // Deallocate device memory
  cudaFree(d_min);
  cudaFree(d_max);
  cudaFree(d_intermediate);
  
  /*
    2) subtract them to find the range
  */
  float range_logLum = max_logLum - min_logLum;
  printf("range_logLum: %f\n", range_logLum);

  /*
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
  */

  // Declare device memory pointers for histogram computation
  unsigned int *d_bins;
  int histogram_bytes = sizeof(unsigned int) * numBins;

  // Allocate device memory
  cudaMalloc((void **) &d_bins, histogram_bytes);

  // Initialize bins in host with value = 0
  unsigned int h_bins[numBins];
  for (int i = 0; i < numBins; i++){
    h_bins[i] = 0;
  }

  // Transfer input arrays from host to device
  cudaMemcpy(d_bins, h_bins, histogram_bytes, cudaMemcpyHostToDevice);

  // Histogram Computation
  // ---------------------
  histogram_atomic<<<blocks, threads>>>(d_bins, d_logLuminance, numBins, min_logLum, range_logLum);

  // Wait for the GPU to finish computations
  cudaDeviceSynchronize();

  /*
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       
  */
  int n_blocks = ceil((float)(numBins)/threads);
  hillis_steele_scan<<<n_blocks,threads>>>(d_bins, numBins);

  // Copy arrays from device to device
  cudaMemcpy(d_cdf, d_bins, histogram_bytes, cudaMemcpyDeviceToDevice);

  // Deallocate device memory
  cudaFree(d_bins);

}
