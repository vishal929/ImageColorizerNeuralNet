
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include <stdio.h>
#include <cuda/std/cmath>


// purpose of this kernel is to do a black and white transformation on a color image, represented by the colorImage array of color pixel values
__global__ void makeImageBlackAndWhite(int* colorR, int* colorG, int* colorB, int* bwImage, int rowDim, int colDim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // looping through each pixel and applying the transformation
    const int maxDim = rowDim * colDim;
    for (int i = tid; i < maxDim; i += blockDim.x * gridDim.x) {
        // storing transformed pixel in bwImage
        bwImage[tid] = (int)(0.21 * ((double)colorR[tid]) + 0.72 * ((double)colorG[tid]) + 0.07 * ((double)colorB[tid]));
    }
}

// calculating the error between a generated color image and the actual color image
// result[0] will hold red error, result[1] will hold green error and result[2] will hold blue error
__global__ void calculateImageError(int* actualR, int* actualG, int* actualB, int* guessedR, int* guessedG, int* guessedB, int* resultR, int* resultG, int* resultB, int rowDim, int colDim) {
    // idea is to store each pixels error in shared memory, and then add up all the error in a reduction -> error is 
    // we will call the kernel with 256 threads per block
    __shared__ int errorR[256];
    __shared__ int errorG[256];
    __shared__ int errorB[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int maxDim = rowDim * colDim;
    // putting error of this pixel into shared memory
    errorR[tid] = abs(actualR[tid] - guessedR[tid]);
    errorG[tid] = abs(actualG[tid] - guessedG[tid]);
    errorB[tid] = abs(actualB[tid] - guessedB[tid]);
    __syncthreads();

    // doing add reduce of shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            // add
            errorR[tid] += errorR[s + tid];
            errorG[tid] += errorG[s + tid];
            errorB[tid] += errorB[s + tid];
        }
        __syncthreads();
    }

    // final atomic add to global memory
    if (tid == 0) {
        // then this thread holds the snared mem reduction, it should atomic add to the result
        atomicAdd(errorR, resultR[0]);
        atomicAdd(errorG, resultG[0]);
        atomicAdd(errorB, resultB[0]);
    }
}
