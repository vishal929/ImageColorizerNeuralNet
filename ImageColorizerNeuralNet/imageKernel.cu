// purpose of this file is to house logic for gpu kernels dealing with image transformation and scaling
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

// purpose of this kernel is to scale an input color image to 4k
__global__ void makeColorImage4K(int* colorR, int* colorG, int* colorB, int* newR, int* newG, int* newB, int rowDim, int colDim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // looping through every pixel and applying the check
    const int resolution = 3840 * 2160;
    for (int i = tid; i < resolution; i += blockDim.x * gridDim.x) {
        int rowNum = i / 3840;
        int colNum = i % 3840;
        if (rowNum >= rowDim || colNum >= colDim) {
            // then this is not part of the original image and should be set to black
            newR[i] = 0;
            newG[i] = 0;
            newB[i] = 0;
        }
        else {
            // then we transfer the color of the original image
            newR[i] = colorR[(rowNum * colDim) + colNum];
            newG[i] = colorG[(rowNum * colDim) + colNum];
            newB[i] = colorB[(rowNum * colDim) + colNum];
        }
    }
}

// wrapper for converting a color image to 4k
void makeColorImage4kWrapper(int* colorR, int* colorG, int* colorB, int* newR, int* newG,  int* newB, int rowDim, int colDim) {
    // first allocating gpu memory
    int* deviceR, * deviceG, * deviceB, * deviceNewR, * deviceNewG, * deviceNewB;
    cudaMalloc(&deviceR, sizeof(int) * rowDim * colDim);
    cudaMalloc(&deviceG, sizeof(int) * rowDim * colDim);
    cudaMalloc(&deviceB, sizeof(int) * rowDim * colDim);
    cudaMalloc(&deviceNewR, sizeof(int) * 3840 * 2160);
    cudaMalloc(&deviceNewG, sizeof(int) * 3840 * 2160);
    cudaMalloc(&deviceNewB, sizeof(int) * 3840 * 2160);
    // copying from host to device
    cudaMemcpy(deviceR, colorR, sizeof(int) * rowDim * colDim, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceG, colorG, sizeof(int) * rowDim * colDim, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, colorB, sizeof(int) * rowDim * colDim, cudaMemcpyHostToDevice);
    // calling the kernel
    makeColorImage4K << <3000, 256>> > (deviceR, deviceG, deviceB, deviceNewR, deviceNewG, deviceNewB, rowDim, colDim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    // copying result to host mem
    cudaMemcpy(newR, deviceNewR, sizeof(int) * 3840 * 2160, cudaMemcpyDeviceToHost);
    cudaMemcpy(newG, deviceNewG, sizeof(int) * 3840 * 2160, cudaMemcpyDeviceToHost);
    cudaMemcpy(newB, deviceNewB, sizeof(int) * 3840 * 2160, cudaMemcpyDeviceToHost);
    // freeing allocated gpu memory
    cudaFree(deviceR);
    cudaFree(deviceG);
    cudaFree(deviceB);
    cudaFree(deviceNewR);
    cudaFree(deviceNewG);
    cudaFree(deviceNewB);
    // now the result arrays should be stored in the newR, newG, and newB pointers
}

// function to scale a black and white image to 4k resolution -> extra pixels are just black
__global__ void makeBlackWhiteImage4K(int* bwValues, int* newBWValues, int rowDim, int colDim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // looping through every pixel and applying the check
    const int resolution = 3840 * 2160;
    for (int i = tid; i < resolution; i += blockDim.x * gridDim.x) {
        int rowNum = i / 3840;
        int colNum = i % 3840;
        if (rowNum >= rowDim || colNum >= colDim) {
            // then this is not part of the original image and should be set to black
            newBWValues[i] = 0;
        }
        else {
            // then we transfer the greyscale value of the original image
            newBWValues[i] = bwValues[(rowNum * colDim) + colNum];
        }
    }
}

// wrapper function for scaling a black and white image to 4k
void makeBlackWhiteImage4KWrapper(int* bwValues, int* newBWValues, int rowDim, int colDim) {
    // first allocating gpu memory
    int* deviceBWValues, * deviceNewBWValues;
    cudaMalloc(&deviceBWValues, sizeof(int) * rowDim * colDim);
    cudaMalloc(&deviceNewBWValues, sizeof(int) * 3840 * 2160);
    // copying from host to device
    cudaMemcpy(deviceBWValues, bwValues, sizeof(int) * rowDim * colDim, cudaMemcpyHostToDevice);
    // calling the kernel
    makeBlackWhiteImage4K << <3000, 256>> > (deviceBWValues, deviceNewBWValues, rowDim, colDim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    // copying result to host mem
    cudaMemcpy(newBWValues, deviceNewBWValues, sizeof(int) * 3840 * 2160, cudaMemcpyDeviceToHost);
    // freeing allocated gpu memory
    cudaFree(deviceNewBWValues);
    cudaFree(deviceBWValues);
    // now the result array should be stored in newBWValues
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
// we also will add features by modifying the data set (i.e if features is set to 2, the squared input will also be included in the patch)
// NOT TESTED
__global__ void getPatches(double* imagePixels, double** imagePatches, int rowDim, int colDim, int patchSize, int features) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int maxIndex = rowDim * colDim;
    for (int i = tid; i < maxIndex; i += gridDim.x * blockDim.x) {
        // filling the associated patch array in global memory
        int row = i / colDim;
        int col = i % colDim;
        for (int j = 0; j < patchSize; j++) {
            for (int z = 0; z < patchSize; z++) {
                int shiftedRow = row - patchSize;
                int shiftedCol = col - patchSize;
                if (shiftedRow < 0 || shiftedRow >= rowDim || shiftedCol < 0 || shiftedCol >= colDim) {
                    // then this pixel is out of bounds, we should color it black in the patch
                    (imagePatches[i])[(patchSize * shiftedRow) + shiftedCol] = 0;
                }
                else {
                    // then this pixel is in the original image, we will copy its value to the patch
                    (imagePatches[i])[(patchSize * shiftedRow) + shiftedCol] = imagePixels[(rowDim*shiftedRow) + shiftedCol];
                }
            }
        }
    }
}

// similar to getPatches, but gets a single patch based on a pixels row and column-> this will help for stochastic gradient descent
__global__ void getPatch(double* imagePixels, double* imagePatch, int rowDim, int colDim, int patchSize, int features, int pixelRow, int pixelCol) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < patchSize * patchSize * features; i += gridDim.x * blockDim.x) {
        // filling the associated patch array in global memory
        int row = (i / patchSize);
        int col = (i % patchSize);
        int shiftedRow = pixelRow - (patchSize/2) + row;
        int shiftedCol = pixelCol - (patchSize/2) + col;
        // getting the feature -> x^1 or x^2 etc.
        int feature = ((i / (patchSize * patchSize)) + 1);
        if (shiftedRow < 0 || shiftedRow >= rowDim || shiftedCol < 0 || shiftedCol >= colDim) {
			// then this pixel is out of bounds, we should color it black in the patch
			imagePatch[(patchSize * row) + col] = 0;
		}
		else {
			// then this pixel is in the original image, we will copy its value to the patch 
			imagePatch[(patchSize * row) + col] = pow(imagePixels[(rowDim*shiftedRow) + shiftedCol], double(feature));
		}
    }
}

//wrapper for the getPatch kernel call
void getPatchWrapper(double* imagePixels, double* imagePatch, int rowDim, int colDim, int patchSize, int features, int pixelRow, int pixelCol) {
    //allocating device memory
    double *deviceImagePixels, *deviceImagePatch;
    cudaMalloc(&deviceImagePixels, sizeof(double) * rowDim * colDim);
    cudaMalloc(&deviceImagePatch, sizeof(double) * patchSize * patchSize * features);
    // copying memory
    cudaMemcpy(deviceImagePixels, imagePixels, sizeof(double) * rowDim * colDim, cudaMemcpyHostToDevice);
    // calling the kernel
    getPatch<<<200, 256>>>(deviceImagePixels, deviceImagePatch, rowDim, colDim, patchSize, features, pixelRow, pixelCol);
    // copying result back to host memory
    cudaMemcpy(imagePatch, deviceImagePatch, sizeof(double) * patchSize * patchSize * features, cudaMemcpyDeviceToHost);
    //freeing gpu memory
    cudaFree(deviceImagePixels);
    cudaFree(deviceImagePatch);
}

// gpu kernel to scale the input pixels to be normalized
__global__ void pixelScale(int* inputPixels, double* outputValues, int rowDim, int colDim) {
    int maxDim = rowDim * colDim;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < maxDim; i += gridDim.x * blockDim.x) {
        // max value is 255, so we just divide the input pixel value by 255
        outputValues[i] = ((double)(inputPixels[i])) / 255;
    }
}

void pixelScaleWrapper(int* inputPixels, double* outputValues, int rowDim, int colDim) {
    int* deviceInputPixels;
    double* deviceOutputValues;
    cudaMalloc(&deviceInputPixels, sizeof(int) * rowDim * colDim);
    cudaMalloc(&deviceOutputValues, sizeof(double) * rowDim * colDim);
    //copying memory
    cudaMemcpy(deviceInputPixels, inputPixels, sizeof(int) * rowDim * colDim, cudaMemcpyHostToDevice);
    // calling kernel
    pixelScale << <200, 256 >> > (deviceInputPixels, deviceOutputValues, rowDim, colDim);
    // copying output back to host memory
    cudaMemcpy(outputValues, deviceOutputValues, sizeof(double) * rowDim * colDim, cudaMemcpyDeviceToHost);
    //freeing gpu memory
    cudaFree(deviceInputPixels);
    cudaFree(deviceOutputValues);
}

__global__ void addFeature(double* inputPixels, double* outputValues, int rowDim, int colDim, int featureNumber) {

}

