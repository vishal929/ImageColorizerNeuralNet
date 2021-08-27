// purpose of this file is to house GPU kernels associated with training and evaluating the neural net
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cuda/std/cmath>

#include "NeuralNet.h"

//big idea for a neural net: 
// input will be greyscale values for every single pixel in the 4k image
// output will be rgbrgbrgb for each pixel in order -> to easily convert to cimg and to easily grab the error

__global__ void inputFunction(double* inputs, int numInputs, layer* inputLayer) {
	// just setting all the inputs to the inputLayer neurons
	int tid = blockIdx.x * gridDim.x + threadIdx.x;
	for (int i = tid; i < numInputs; i += blockDim.x * gridDim.x) {
		((inputLayer->neurons)[i])->input = inputs[i];
	}
}

// evaluating inputs for every neuron in a layer and setting the second layer output
__global__ void evaluateLayer(layer* firstLayer, layer* secondLayer) {
	int tid = blockIdx.x * gridDim.x + threadIdx.x;
	for (int i = tid; i < firstLayer->numNeurons; i += gridDim.x * blockDim.x) {
		double output = (firstLayer->function)((firstLayer->neurons)[i]->input);
		// adjusting second layer
		for (int j = tid; j < secondLayer->numNeurons; j += gridDim.x * blockDim.x) {
			atomicAdd(&(((secondLayer->neurons)[j])->input), output*(firstLayer->neurons[i]->weights[j]));
		}
	}
}


