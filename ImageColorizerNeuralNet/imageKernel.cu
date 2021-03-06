// purpose of this file is to house logic for gpu kernels dealing with image transformation and scaling
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "cuda/std/cmath"
#include "cudaErrorHandler.cuh"


// purpose of this kernel is to do a black and white transformation on a color image, represented by the colorImage array of color pixel values
__global__ void makeImageBlackAndWhite(double* colorR, double* colorG, double* colorB, double* bwImage, int rowDim, int colDim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // looping through each pixel and applying the transformation
    const int maxDim = rowDim * colDim;
    for (int i = tid; i < maxDim; i += blockDim.x * gridDim.x) {
        // storing transformed pixel in bwImage
        bwImage[i] = (0.21 * ((colorR[i])) + 0.72 * ((colorG[i])) + 0.07 * ((colorB[i])));
    }
}

// function that just wraps cpu logic with the kernel call for black and white image
// IMPORTANT: all inputted parameters should be allocated before calling the function wrapper
void makeImageBlackAndWhiteWrapper(double* colorR, double* colorG, double* colorB, double* bwImage, int rowDim, int colDim) {
    // first allocating gpu memory
    double* deviceR, * deviceG, * deviceB, * deviceBWImage;
    cudaErrorCheck(cudaMalloc(&deviceR, sizeof(double) * rowDim * colDim));
    
    cudaErrorCheck(cudaMalloc(&deviceG, sizeof(double) * rowDim * colDim));
    cudaErrorCheck(cudaMalloc(&deviceB, sizeof(double) * rowDim * colDim));
    

    cudaErrorCheck(cudaMalloc(&deviceBWImage, sizeof(double) * rowDim * colDim));
    // copying from host to device
    cudaErrorCheck(cudaMemcpy(deviceR, colorR, sizeof(double) * rowDim * colDim, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(deviceG, colorG, sizeof(double) * rowDim * colDim, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(deviceB, colorB, sizeof(double) * rowDim * colDim, cudaMemcpyHostToDevice));
    // calling the kernel (for a 1080p image, we can launch 8100 blocks with 256 threads each)
    makeImageBlackAndWhite<< <20, 512>> >(deviceR, deviceG, deviceB, deviceBWImage, rowDim, colDim);
    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
        printf("error with make image black and white wrapper %s\n", cudaGetErrorString(lastError));
    }
    
    // copying result to host mem
    cudaErrorCheck(cudaMemcpy(bwImage, deviceBWImage, sizeof(double) * rowDim * colDim, cudaMemcpyDeviceToHost));
    
    // freeing allocated gpu memory
    cudaErrorCheck(cudaFree(deviceR));
    cudaErrorCheck(cudaFree(deviceG));
    cudaErrorCheck(cudaFree(deviceB));
    cudaErrorCheck(cudaFree(deviceBWImage));
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
    cudaErrorCheck(cudaMalloc(&deviceR, sizeof(int) * rowDim * colDim));
    
    cudaErrorCheck(cudaMalloc(&deviceG, sizeof(int) * rowDim * colDim));
    
    cudaErrorCheck(cudaMalloc(&deviceB, sizeof(int) * rowDim * colDim));
    
    cudaErrorCheck(cudaMalloc(&deviceNewR, sizeof(int) * 3840 * 2160));
    
    cudaErrorCheck(cudaMalloc(&deviceNewG, sizeof(int) * 3840 * 2160));
    
    cudaErrorCheck(cudaMalloc(&deviceNewB, sizeof(int) * 3840 * 2160));
    // copying from host to device
    cudaErrorCheck(cudaMemcpy(deviceR, colorR, sizeof(int) * rowDim * colDim, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(deviceG, colorG, sizeof(int) * rowDim * colDim, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(deviceB, colorB, sizeof(int) * rowDim * colDim, cudaMemcpyHostToDevice));
    // calling the kernel
    makeColorImage4K << <20, 512>> > (deviceR, deviceG, deviceB, deviceNewR, deviceNewG, deviceNewB, rowDim, colDim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error with making color image 4k wrapper: %s\n", cudaGetErrorString(err));
    }
    // copying result to host mem
    cudaErrorCheck(cudaMemcpy(newR, deviceNewR, sizeof(int) * 3840 * 2160, cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaMemcpy(newG, deviceNewG, sizeof(int) * 3840 * 2160, cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaMemcpy(newB, deviceNewB, sizeof(int) * 3840 * 2160, cudaMemcpyDeviceToHost));
    // freeing allocated gpu memory
    cudaErrorCheck(cudaFree(deviceR));
    cudaErrorCheck(cudaFree(deviceG));
    cudaErrorCheck(cudaFree(deviceB));
    cudaErrorCheck(cudaFree(deviceNewR));
    cudaErrorCheck(cudaFree(deviceNewG));
    cudaErrorCheck(cudaFree(deviceNewB));
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
    cudaErrorCheck(cudaMalloc(&deviceBWValues, sizeof(int) * rowDim * colDim));
    cudaErrorCheck(cudaMalloc(&deviceNewBWValues, sizeof(int) * 3840 * 2160));
    // copying from host to device
    cudaErrorCheck(cudaMemcpy(deviceBWValues, bwValues, sizeof(int) * rowDim * colDim, cudaMemcpyHostToDevice));
    // calling the kernel
    makeBlackWhiteImage4K << <20, 512>> > (deviceBWValues, deviceNewBWValues, rowDim, colDim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error with make black and white image 4k wrapper: %s\n", cudaGetErrorString(err));
    }
    // copying result to host mem
    cudaErrorCheck(cudaMemcpy(newBWValues, deviceNewBWValues, sizeof(int) * 3840 * 2160, cudaMemcpyDeviceToHost));
    // freeing allocated gpu memory
    cudaErrorCheck(cudaFree(deviceNewBWValues));
    cudaErrorCheck(cudaFree(deviceBWValues));
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
    for (int i = tid;i < patchSize * patchSize * features; i += gridDim.x * blockDim.x) {
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
            // only using standard features for now
			imagePatch[(patchSize * row) + col] = pow(imagePixels[(colDim*shiftedRow) + shiftedCol], features);
		}
    }
}

//wrapper for the getPatch kernel call
void getPatchWrapper(double* imagePixels, double* imagePatch, int rowDim, int colDim, int patchSize, int features, int pixelRow, int pixelCol) {
    //allocating device memory
    double *deviceImagePixels, *deviceImagePatch;
    cudaErrorCheck(cudaMalloc(&deviceImagePixels, sizeof(double) * rowDim * colDim));
    cudaErrorCheck(cudaMalloc(&deviceImagePatch, sizeof(double) * patchSize * patchSize * features));
    // copying memory
    cudaErrorCheck(cudaMemcpy(deviceImagePixels, imagePixels, sizeof(double) * rowDim * colDim, cudaMemcpyHostToDevice));
    // calling the kernel
    getPatch<<<20, 512>>>(deviceImagePixels, deviceImagePatch, rowDim, colDim, patchSize, features, pixelRow, pixelCol);
    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
        printf("error with patch wrapper %s\n", cudaGetErrorString(lastError));
    }
    cudaErrorCheck(cudaMemcpy(imagePatch, deviceImagePatch, sizeof(double) * patchSize * patchSize*features , cudaMemcpyDeviceToHost));
    //freeing gpu memory
    cudaFree(deviceImagePixels);
    cudaFree(deviceImagePatch);
}

__global__ void getSquare(double* inputPixels, double* squarePixels, int squareSideLength, int rowDim, int colDim, int pixelRow, int pixelCol) {
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    for (int i = tidx;i < squareSideLength;i += gridDim.x * blockDim.x) {
        for (int j =  tidy;j <  squareSideLength;j += gridDim.y * blockDim.y) {
            int adjustedRow = i - (squareSideLength / 2) + pixelRow;
            int adjustedCol = j - (squareSideLength / 2) + pixelCol;
            if (adjustedRow < 0 || adjustedRow >= rowDim || adjustedCol <0 || adjustedCol >=colDim) {
                // then this should just be black in the square
                squarePixels[(i* squareSideLength) + j] = 0;
            }
            else {
                // then this should just propogate the input pixel
                squarePixels[(i* squareSideLength) + j] = inputPixels[(adjustedRow * colDim) + adjustedCol];
            }
        }
    }
}



// gets all the squares around pixels for the number of squares requested
// each thread handles a single square around a pixel
// this is more for evaluating the neural net, because we can just evaluate all the squares at once (or as many as possible), so it makes sense to grab as many squares as possible
__global__ void getSquares(double* inputPixels, double* squares, int squareSideLength, int rowDim, int colDim, int numSquares, int startPixel) {
   int tidx = blockDim.x * blockIdx.x + threadIdx.x;
   for (int i = tidx; i < numSquares; i += gridDim.x * blockDim.x) {
       for (int j = 0;j < squareSideLength;j++) {
           for (int z = 0; z < squareSideLength;z++) {
                int adjustedRow = j- (squareSideLength/2) + ((startPixel + i)/colDim);
                int adjustedCol = z- (squareSideLength/2) + ((startPixel + i) % colDim);
                if (adjustedRow < 0 || adjustedRow >= rowDim || adjustedCol < 0 || adjustedCol >= colDim) {
                    // then this should just be black in the square
                    squares[(((j*squareSideLength)+z) * numSquares) + i] = 0;
                }
                else {
                    // then this should just propogate the input pixel
                    squares[(((j*squareSideLength)+z) * numSquares) + i] = inputPixels[(adjustedRow * colDim) + adjustedCol];
                }
           }
       }
       
   }
}

void getSquaresWrapper(double* inputPixels, double* squares, int squareSideLength, int rowDim, int colDim, int numSquares, int startPixel) {
    double* devicePixels, * deviceSquares;
    /*
    size_t freeMem;
    size_t totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("total memory of gpu: %ul\n",totalMem);
    printf( "total free memory of gpu before square allocation: %ul\n",freeMem);
    */
    cudaErrorCheck(cudaMalloc(&devicePixels, sizeof(double) * rowDim * colDim));
    cudaErrorCheck(cudaMalloc(&deviceSquares, sizeof(double) * squareSideLength * squareSideLength * numSquares));

    /*
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("total memory of gpu: %ul\n",totalMem);
    printf( "total free memory of gpu after square allocation: %ul\n",freeMem);
    */

    cudaErrorCheck(cudaMemcpy(devicePixels, inputPixels, sizeof(double) * rowDim * colDim, cudaMemcpyHostToDevice));
    
    //calling kernel
    getSquares<<<20,512>>>(devicePixels, deviceSquares, squareSideLength, rowDim, colDim, numSquares, startPixel);

    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
        printf("error with get get squares wrapper %s\n", cudaGetErrorString(lastError));
    }

    //copying to host
    cudaErrorCheck(cudaMemcpy(squares, deviceSquares, sizeof(double) * squareSideLength * squareSideLength * numSquares, cudaMemcpyDeviceToHost));

    // freeing memory
    cudaErrorCheck(cudaFree(devicePixels));
    cudaErrorCheck(cudaFree(deviceSquares));
}

void getSquareCPU(double* inputPixels, double* squarePixels, int squareSideLength, int rowDim, int colDim, int pixelRow, int pixelCol) {
    for (int i = 0;i < squareSideLength;i++) {
        for (int j = 0;j < squareSideLength;j++) {
            int adjustedRow = i - (squareSideLength / 2) + pixelRow;
            int adjustedCol = j - (squareSideLength / 2) + pixelCol;
            if (adjustedRow < 0 || adjustedRow >= rowDim || adjustedCol <0 || adjustedCol >=colDim) {
                // then this should just be black in the square
                squarePixels[(i* squareSideLength) + j] = 0;
            }
            else {
                // then this should just propogate the input pixel
                squarePixels[(i* squareSideLength) + j] = inputPixels[(adjustedRow * colDim) + adjustedCol];
            }
        }
    }
}

//wrapper for getting the square kernel
void getSquareWrapper(double* inputPixels, double* squarePixels, int squareSideLength, int rowDim, int colDim, int pixelRow, int pixelCol) {
    double* devicePixels, * deviceSquarePixels;
    /*
    size_t freeMem;
    size_t totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("total memory of gpu: %ul\n",totalMem);
    printf( "total free memory of gpu before square allocation: %ul\n",freeMem);
    */
    cudaErrorCheck(cudaMalloc(&devicePixels, sizeof(double) * rowDim * colDim));
    cudaErrorCheck(cudaMalloc(&deviceSquarePixels, sizeof(double) * squareSideLength * squareSideLength));

    /*
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("total memory of gpu: %ul\n",totalMem);
    printf( "total free memory of gpu after square allocation: %ul\n",freeMem);
    */

    cudaErrorCheck(cudaMemcpy(devicePixels, inputPixels, sizeof(double) * rowDim * colDim, cudaMemcpyHostToDevice));
    
    dim3 blockShape(16, 16);
    dim3 gridShape(4, 4);
    //calling kernel
    getSquare<<<gridShape, blockShape>>>(devicePixels, deviceSquarePixels, squareSideLength, rowDim, colDim, pixelRow, pixelCol);
    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
        printf("error with get get square wrapper %s\n", cudaGetErrorString(lastError));
    }

    //copying to host
    cudaErrorCheck(cudaMemcpy(squarePixels, deviceSquarePixels, sizeof(double) * squareSideLength * squareSideLength, cudaMemcpyDeviceToHost));

    // freeing memory
    cudaErrorCheck(cudaFree(devicePixels));
    cudaErrorCheck(cudaFree(deviceSquarePixels));
}

// gpu kernel to scale the input pixels to be normalized
__global__ void pixelScale(double* inputPixels, double* outputValues, int rowDim, int colDim,double scalar) {
    int maxDim = rowDim * colDim;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < maxDim; i += gridDim.x * blockDim.x) {
        // max value is 255, so we just divide the input pixel value by 255
        //outputValues[i] = ((double)(inputPixels[i])) / (255.0 * 62500.0);
        outputValues[i] = ((inputPixels[i])) / (scalar);
    }
}

void pixelScaleWrapper(double* inputPixels, double* outputValues, int rowDim, int colDim, double scalar) {
    double* deviceInputPixels;
    double* deviceOutputValues;
    cudaErrorCheck(cudaMalloc(&deviceInputPixels, sizeof(double) * rowDim * colDim));
    cudaErrorCheck(cudaMalloc(&deviceOutputValues, sizeof(double) * rowDim * colDim));
    //copying memory
    cudaErrorCheck(cudaMemcpy(deviceInputPixels, inputPixels, sizeof(double) * rowDim * colDim, cudaMemcpyHostToDevice));
    // calling kernel
    pixelScale << <20, 512 >> > (deviceInputPixels, deviceOutputValues, rowDim, colDim, scalar);
    cudaError_t lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
        printf("error with get patch wrapper %s\n", cudaGetErrorString(lastError));
    }
    // copying output back to host memory
    cudaErrorCheck(cudaMemcpy(outputValues, deviceOutputValues, sizeof(double) * rowDim * colDim, cudaMemcpyDeviceToHost));
    //freeing gpu memory
    cudaErrorCheck(cudaFree(deviceInputPixels));
    cudaErrorCheck(cudaFree(deviceOutputValues));
}

__global__ void addFeature(double* inputPixels, double* outputValues, int rowDim, int colDim, int featureNumber) {

}

//testing the get squares function
// the idea is that we get the first 100 or so squares from a buffer and then verify that these are in fact correct with cpu code
void getSquaresTest(double* data, int rowDim, int colDim, int numSquaresToCheck, int squareLength) {
    double* gpuResult = (double*)malloc(sizeof(double) * squareLength * squareLength * numSquaresToCheck);
    double* cpuResult = (double*)malloc(sizeof(double) * squareLength * squareLength * numSquaresToCheck);

    getSquaresWrapper(data, gpuResult, squareLength, rowDim, colDim, numSquaresToCheck, 0);

    // getting squares for cpu
    for (int i = 0;i < numSquaresToCheck;i++) {
        int row = i / colDim;
        int col = i % colDim;
        double* intermedSquare = (double*)malloc(sizeof(double) * squareLength * squareLength);
        // filling square
        getSquareCPU(data, intermedSquare, squareLength, rowDim, colDim, row, col);
        // filling the cpuResult buffer
        for (int j = 0;j < squareLength * squareLength; j++) {
            cpuResult[(j * numSquaresToCheck) + i] = intermedSquare[j];
        }
        free(intermedSquare);
    }

    //asserting equality
    for (int i = 0;i < squareLength * squareLength * numSquaresToCheck;i++) {
        if (gpuResult[i] != cpuResult[i]) {
            printf("VERY BAD ERROR WITH GRABBING SQUARES!!!! -> INCONSISTENT FOR MULTI-GRAB!\n");
        }
        /*
        else {
            printf("all gud!\n");
        }
        */
    }

    free(gpuResult);
    free(cpuResult);
}

