
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
        bwImage[i] = (int)(0.21 * ((double)(colorR[i])) + 0.72 * ((double)(colorG[i])) + 0.07 * ((double)(colorB[i])));
    }
}

// function that just wraps cpu logic with the kernel call for black and white image
// IMPORTANT: all inputted parameters should be allocated before calling the function wrapper
void makeImageBlackAndWhiteWrapper(int* colorR, int* colorG, int* colorB, int* bwImage, int rowDim, int colDim) {
    // first allocating gpu memory
    int* deviceR, * deviceG, * deviceB, * deviceBWImage;
    cudaMalloc(&deviceR, sizeof(int) * rowDim * colDim);
    cudaMalloc(&deviceG, sizeof(int) * rowDim * colDim);
    cudaMalloc(&deviceB, sizeof(int) * rowDim * colDim);
    cudaMalloc(&deviceBWImage, sizeof(int) * rowDim * colDim);
    // copying from host to device
    cudaMemcpy(deviceR, colorR, sizeof(int) * rowDim * colDim, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceG, colorG, sizeof(int) * rowDim * colDim, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, colorB, sizeof(int) * rowDim * colDim, cudaMemcpyHostToDevice);
    // calling the kernel (for a 1080p image, we can launch 8100 blocks with 256 threads each)
    makeImageBlackAndWhite<< <3000, 256 >> >(deviceR, deviceG, deviceB, deviceBWImage, rowDim, colDim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    // copying result to host mem
    cudaMemcpy(bwImage, deviceBWImage, sizeof(int) * rowDim * colDim, cudaMemcpyDeviceToHost);
    // freeing allocated gpu memory
    cudaFree(deviceR);
    cudaFree(deviceG);
    cudaFree(deviceB);
    cudaFree(deviceBWImage);
    // now the result black and white image should be stored in the host bwImage pointer
}

// gpu kernel to do a dot product between two vectors -> this will be used in evaluating weights for our functions in the neural net
// although a cpu can probably handle vector dots for relatively decent vector sizes during gradient descent training
__global__ void vectorDot(double* weights, double* input, double* result, int length) {
   // shared memory for add-reduce
    __shared__ double toReduce[256];
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // placing a product into shared memory 
    for (int i = tid; i < length; i += gridDim.x * blockDim.x) {
        toReduce[i] = weights[i] * input[i];
    }
    // need to sync threads before add reduce
    __syncthreads();
    // add reduce and then atomic add into global memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            toReduce[tid] += toReduce[tid + s];
        }
        __syncthreads();
    }
    // atomic add into global memory
    if (tid == 0) {         
        atomicAdd(toReduce, result[0]);
    }
}

//idea here is to associate each pixel with some patch -> will aid in feature detection
// if the pixel is on an edge, we just fill the rest of the patch with black
__global__ void getPatches(double* imagePixels, double** imagePatches, int rowDim, int colDim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = rowDim * colDim;
    for (int i = tid; i < maxIndex; i += gridDim.x * blockDim.x) {
        // filling the associated patch array in global memory
        int row = tid / colDim;
        int col = tid - (row * colDim);
        for (int j = row-30;j != row+30;j++) {
            for (int z = col-30; z != col+30; z++) {
                // 31x31 patch of pixels will be considered for now
                if (j<0 || j>= rowDim || z<0 || z>=colDim ) {
                    //then the pixel in the loop is out of bounds, we will stick in black scaled pixel
                    imagePatches[i][(31*(j-row+30)) + (z-col+30)] = 1;
                }
                else {
                    // then we copy the pixel value
                    imagePatches[i][(31*(j-row+30)) + (z-col+30)] = imagePixels[(rowDim * j) + z];
                }
            }
        }
    }
}


// idea here is to take our input, apply the weights to every input pixel, and then modify the result buffers, we will use sigmoid for now
__global__ void evaluateModel(double** inputPatches, double* weightsR, double* weightsG, double* weightsB, int numWeights, int* resultR, int* resultG, int* resultB, int rowDim, int colDim) {   
    // each thread handles a single pixel scaled value 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid;i < rowDim * colDim; i += gridDim.x * blockDim.x) {
        //
    }
}

// gpu kernel to scale the input pixels to be normalized
__global__ void pixelScale(int* inputPixels, double* outputValues, int rowDim, int colDim) {
    int maxDim = rowDim * colDim;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < maxDim; i += gridDim.x * blockDim.x) {
        // max value is 255, so we just divide the input pixel value by 255
        outputValues[i] = ((double)(inputPixels[i])) / 255;
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
