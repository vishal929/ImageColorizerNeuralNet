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
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numBiases; i += gridDim.x * blockDim.x) {
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

__global__ void trainingHelper(net* toTrain, double** currLayerOutput, double** nextDerivatives, double learningRate) {
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;
	for (int i = toTrain->numLayers-1; i >=0; i--) {
		layer* toConsider = toTrain->neuralLayers[i];
		for (int j = tidX; j < toConsider->numNeuronsNextLayer; j += blockDim.x * gridDim.x) {
			for (int z = tidY; z < toConsider->numNeuronsCurrentLayer; z += blockDim.y * gridDim.y) {
				toConsider->weightAdjustments[(j * toConsider->numNeuronsCurrentLayer) + z] = nextDerivatives[i][j] * currLayerOutput[i][j] * (1 - currLayerOutput[i][j]) * toConsider->neuronInputs[z];
				if (i != 0) {
					atomicAdd(&(nextDerivatives[i - 1][z]) , nextDerivatives[i][j] * currLayerOutput[i][j] * (1 - currLayerOutput[i][j]) * toConsider->weightMatrix[(j * toConsider->numNeuronsCurrentLayer) + z]);
				}
			}
			//adjusting the bias while we can
			if (tidX == 0 && tidY == 0) {
				toConsider->biases[j] -= learningRate * nextDerivatives[i][j] * currLayerOutput[i][j] * (1 - currLayerOutput[i][j]);
			}
		}
		//syncing threads before moving onto the next layer of backpropogation
		__syncthreads();
	}
	__syncthreads();

	// after syncing threads doing adjustments for all the weights
	for (int i = toTrain->numLayers-1; i >=0; i--) {
		layer* toConsider = toTrain->neuralLayers[i];
		for (int j = tidX; j < toConsider->numNeuronsNextLayer; j += blockDim.x * gridDim.x) {
			for (int z = tidY; z < toConsider->numNeuronsCurrentLayer; z += blockDim.y * gridDim.y) {
				// doing adjustment
				toConsider->weightMatrix[(j * toConsider->numNeuronsCurrentLayer) + z] -= learningRate * toConsider->weightAdjustments[(j * toConsider->numNeuronsCurrentLayer)];
			}
		}
	}

	// now all the weights are changed, we will memcpy the weights in the helper

}

void trainingHelperWrapper(net* toTrain, double* netOutput, double actualR, double actualG, double actualB, double learningRate) {
	double derivatives[3];
	double dEdR = -(((double)(actualR / 255)) - netOutput[0]);
	double dEdG = -(((double)(actualG / 255)) - netOutput[1]);
	double dEdB = -(((double)(actualB / 255)) - netOutput[2]);
	derivatives[0] = dEdR;
	derivatives[1] = dEdG;
	derivatives[2] = dEdB;
	// end of getting initial partial derivatives
	net* deviceNet;
	double **currLayerOutput, ** nextDerivatives;
	// allocating and copying the struct to the gpu
	cudaMalloc(&deviceNet, sizeof(net));
	cudaMemcpy(deviceNet, toTrain, sizeof(net), cudaMemcpyHostToDevice);
	// allocating inner fields of the neural net
	//cudaMalloc(&(deviceNet->inputs), sizeof(double) * toTrain->numInputs);
	//cudaMemcpy(deviceNet->inputs, toTrain->inputs, sizeof(double) * toTrain->numInputs, cudaMemcpyHostToDevice);

	//allocating stored outputs and learning rates we will need
	cudaMalloc(&(currLayerOutput), sizeof(double*) * toTrain->numLayers);
	cudaMalloc(&(nextDerivatives), sizeof(double*) * toTrain->numLayers);
	// allocating layers
	cudaMalloc(&(deviceNet->neuralLayers), sizeof(layer*) * toTrain->numLayers);
	for (int i = 0; i < toTrain->numLayers; i++) {
		cudaMalloc(&(deviceNet->neuralLayers[i]), sizeof(layer));
		cudaMemcpy(deviceNet->neuralLayers[i], toTrain->neuralLayers[i], sizeof(layer), cudaMemcpyHostToDevice);
		// copying inputs, outputs, biases, and allocating adjustments
		cudaMalloc(&(deviceNet->neuralLayers[i]->neuronInputs), sizeof(double) * toTrain->neuralLayers[i]->numNeuronsCurrentLayer);
		cudaMemcpy(deviceNet->neuralLayers[i]->neuronInputs, toTrain->neuralLayers[i]->neuronInputs, sizeof(double) * toTrain->neuralLayers[i]->numNeuronsCurrentLayer, cudaMemcpyHostToDevice);
		cudaMalloc(&(deviceNet->neuralLayers[i]->weightMatrix), sizeof(double) * toTrain->neuralLayers[i]->numNeuronsCurrentLayer * toTrain->neuralLayers[i]->numNeuronsNextLayer);
		cudaMemcpy(deviceNet->neuralLayers[i]->weightMatrix, toTrain->neuralLayers[i]->weightMatrix, sizeof(double) * toTrain->neuralLayers[i]->numNeuronsCurrentLayer * toTrain->neuralLayers[i]->numNeuronsNextLayer, cudaMemcpyHostToDevice);
		cudaMalloc(&(deviceNet->neuralLayers[i]->biases), sizeof(double) * toTrain->neuralLayers[i]->numNeuronsNextLayer);
		cudaMemcpy(deviceNet->neuralLayers[i]->biases, toTrain->neuralLayers[i]->biases, sizeof(double) * toTrain->neuralLayers[i]->numNeuronsNextLayer, cudaMemcpyHostToDevice);
		// allocating adjustments for gpu to fill
		cudaMalloc(&(deviceNet->neuralLayers[i]->weightAdjustments), sizeof(double) * toTrain->neuralLayers[i]->numNeuronsCurrentLayer * toTrain->neuralLayers[i]->numNeuronsNextLayer);
		// allocating special memory for training
		cudaMalloc(&(currLayerOutput[i]), sizeof(double) * toTrain->neuralLayers[i]->numNeuronsNextLayer);
		cudaMalloc(&(nextDerivatives[i]), sizeof(double) * toTrain->neuralLayers[i]->numNeuronsNextLayer);
		if (i==(toTrain->numLayers)-1){
			cudaMemcpy(currLayerOutput[i], netOutput, sizeof(double) * 3, cudaMemcpyHostToDevice);
			cudaMemcpy(nextDerivatives[i], derivatives, sizeof(double) * 3, cudaMemcpyHostToDevice);
		} else {
			cudaMemcpy(currLayerOutput[i], toTrain->neuralLayers[i + 1]->neuronInputs, sizeof(double) * toTrain->neuralLayers[i + 1]->numNeuronsCurrentLayer, cudaMemcpyHostToDevice);
			cudaMemset(nextDerivatives[i], 0, sizeof(double) * toTrain->neuralLayers[i]->numNeuronsNextLayer);
		}
	}

	// calling the kernel


	trainingHelper << <200, 256 >> > (deviceNet, currLayerOutput, nextDerivatives, learningRate);

	// freeing memory and copying the updated weights back to the CPU struct -> do not need anything else 
	for (int i = 0; i < toTrain->numLayers; i++) {
		// copying updated weights back to cpu struct and then freeing inner objects
		cudaMemcpy(deviceNet->neuralLayers[i]->weightMatrix, toTrain->neuralLayers[i]->weightMatrix, sizeof(double) * toTrain->neuralLayers[i]->numNeuronsCurrentLayer * toTrain->neuralLayers[i]->numNeuronsNextLayer, cudaMemcpyDeviceToHost);

		cudaFree(nextDerivatives[i]);
		cudaFree(currLayerOutput[i]);

		cudaFree(deviceNet->neuralLayers[i]->weightAdjustments);
		cudaFree(deviceNet->neuralLayers[i]->biases);
		cudaFree(deviceNet->neuralLayers[i]->weightMatrix);
		cudaFree(deviceNet->neuralLayers[i]->neuronInputs);

		cudaFree(deviceNet->neuralLayers[i]);
	}
	
	cudaFree(nextDerivatives);
	cudaFree(currLayerOutput);

	
	// freeing outer net
	cudaFree(deviceNet->neuralLayers);
	cudaFree(deviceNet);
}

// applies relu activation function to results
__global__ void reluResults(double* inputs, int numInputs) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numInputs; i += gridDim.x * blockDim.x) {
		inputs[i] = fmaxf(0, inputs[i]);
	}
}

// applies sigmoid activation function to results
__global__ void sigmoidResults(double* inputs, int numInputs) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numInputs; i += gridDim.x * blockDim.x) {
		inputs[i] = 1 / (1 + _CUDA_CMATH exp(-inputs[i]));
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

