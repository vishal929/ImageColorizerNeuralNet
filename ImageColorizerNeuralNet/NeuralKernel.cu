// purpose of this file is to house GPU kernels associated with training and evaluating the neural net
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cuda/std/cmath>
#include <cublas.h>

#include "NeuralNet.h"

//big idea for a neural net: 
// input will be greyscale values for every single pixel in the 4k image
// output will be rgbrgbrgb for each pixel in order -> to easily convert to cimg and to easily grab the error

// idea is 4 layers (input layer, 2 hidden layer, output layer with 3 output neurons) for now with 100 neurons and ReLu activation function layer into the Sigmoid activation function layer which feeds into output
// we will try 3 layers with only sigmoid now, and if that isnt enough we will add a relu layer before and see if that helps

// evaluating inputs for every neuron in a layer and setting the second layer output
// this is accomplished with matrix multiplication

// we will use cuBLAS NVIDIA api for fast matrix multiplication 
void layerMultiplicationWrapper(double* weights, double* inputs, double* biases, double* output, int numNeuronsNextLayer, int numNeuronsCurrentLayer) {
	//wrapping multiplication with cublas	
	double* deviceWeights, * deviceInputs, * deviceBiases;
	cudaMalloc(&deviceWeights, sizeof(double) * numNeuronsNextLayer * numNeuronsCurrentLayer);
	cudaMalloc(&deviceInputs, sizeof(double) * numNeuronsCurrentLayer);	
	cudaMalloc(&deviceBiases, sizeof(double) * numNeuronsNextLayer);
	
	// copying host memory to device
	cudaMemcpy(deviceWeights, weights, sizeof(double) * numNeuronsCurrentLayer*numNeuronsNextLayer, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInputs, inputs, sizeof(double) * numNeuronsCurrentLayer, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceBiases, biases, sizeof(double) * numNeuronsNextLayer, cudaMemcpyHostToDevice);

	//calling cublas matrix multiply and adding biases vector (this does deviceWeights*deviceInputs + biasVector) and stores the result in the bias vector
	cublasDgemm(CUBLAS_OP_N, CUBLAS_OP_N, numNeuronsNextLayer, 1, numNeuronsCurrentLayer, 1, deviceWeights, numNeuronsNextLayer, deviceInputs, numNeuronsCurrentLayer, 1, deviceBiases, numNeuronsNextLayer);

	// copying result of multiplication and addition back to output host memory
	cudaMemcpy(output, deviceBiases, sizeof(double) * numNeuronsNextLayer, cudaMemcpyDeviceToHost);

	//freeing device memory
	cudaFree(deviceWeights);
	cudaFree(deviceInputs);
	cudaFree(deviceBiases);
}

// will need to add biases to matrix results if any -> we have as many biases as results
__global__ void biasAdd(double* results,double* biases, int numBiases) {
	for (int i = blockIdx.x * gridDim.x + threadIdx.x; i < numBiases; i += gridDim.x * blockDim.x) {
		results[i] += biases[i];
	}
}

void biasAddWrapper(double* results, double* biases, int numBiases) {
	double* deviceResults, * deviceBiases;
	cudaMalloc(&deviceResults, sizeof(double) * numBiases);
	cudaMalloc(&deviceBiases, sizeof(double) * numBiases);

	cudaMemcpy(deviceResults, results, sizeof(double) * numBiases, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceBiases, biases, sizeof(double) * numBiases, cudaMemcpyHostToDevice);

	// calling kernel
	biasAdd << <200, 256 >> > (deviceResults, deviceBiases, numBiases);
	
	// copying output
	cudaMemcpy(results, deviceResults, sizeof(double) * numBiases, cudaMemcpyDeviceToHost);

	// freeing GPU memory
	cudaFree(deviceResults);
	cudaFree(deviceBiases);
}

// applies relu activation function to results
__global__ void reluResults(double* inputs, int numInputs) {
	for (int i = blockIdx.x * gridDim.x + threadIdx.x; i < numInputs; i += gridDim.x * blockDim.x) {
		inputs[i] = fmaxf(0, inputs[i]);
	}
}

// applies sigmoid activation function to results
__global__ void sigmoidResults(double* inputs, int numInputs) {
	for (int i = blockIdx.x * gridDim.x + threadIdx.x; i < numInputs; i += gridDim.x * blockDim.x) {
		inputs[i] = 1 / (1 + exp(-inputs[i]));
	}
}

void sigmoidWrapper(double* inputs, int numInputs) {
	double* deviceInputs;
	cudaMalloc(&deviceInputs, sizeof(double) * numInputs);

	cudaMemcpy(deviceInputs, inputs, sizeof(double) * numInputs, cudaMemcpyHostToDevice);

	sigmoidResults << <200, 256 >> > (deviceInputs, numInputs);
	//copying back to host and freeing memory
	cudaMemcpy(inputs, deviceInputs, sizeof(double) * numInputs, cudaMemcpyDeviceToHost);
	cudaFree(deviceInputs);
}

