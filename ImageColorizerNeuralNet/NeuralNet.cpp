// file serves as a container for logic associated with neural net specifics -> allows building and training the model on CPU
// The gpu specific logic will be implemented later -> the gpu can use the structs, but the function will need to be adjusted on the CUDA side

#include <math.h>
#include "NeuralNet.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
using namespace std;

double inputFunction(double input);



// function for an input layer
double inputFunction(double input) {
	return input;
}

// function for ReLu 
double reluFunction(double input) {
	return fmax(0, input);
}

// function for matrix multiplication for layer weights calculation
double weightCalculation(double* inputs, double* weights, int inputsRows, int weightsColumns) {

}

// initializing a neural net based on weights

// initializing a neural net based on random weights (before training)

// function to write weights back to txt file (for later loading/testing)
// each layer will have a different txt file with weights

// we may want to load a model from weights after training sessions so we do not "lose" progress
net* loadNetFromWeights() {

}

//we will want to periodically write back weights when training the model
void writeWeights(net* neuralNet) {
	for (int i = 0; i < neuralNet->numLayers; i++) {
		// writing the weights of each layer to a file in matrix form
		// the biases will be written after the matrix after a newline
		/*
			Something like: 
			numNeurons numNeuronsInNextLayer
			x x x
			x x x
			x x x 
			START OF BIASES:
			x
			x
			x
			x
		*/
		layer* currLayer = neuralNet->neuralLayers[i];
		char* basename = (char*) malloc(sizeof(char) * 13);
		if (basename == NULL) {
			cout << "malloc for basename layer: " << i << " failed!\n";
		}
		char layerNum = '0' + i;
		basename[0] = layerNum;
		memcpy(basename + 1, &"weights.txt\0", sizeof(char)*12);

		FILE* layer_weights = fopen(basename, "w");
		// writing the number of neurons in this layer and then the next layer
		fprintf(layer_weights, "%d %d\n", currLayer->numNeuronsCurrentLayer, currLayer->numNeuronsNextLayer);
		// writing the matrix format of the weights in this layer
		for (int j = 0; j < currLayer->numNeuronsNextLayer; j++) {
			for (int z = 0; z < currLayer->numNeuronsCurrentLayer; z++) {
				if (z != currLayer->numNeuronsCurrentLayer-1) {
					fprintf(layer_weights,"%lf ", currLayer->weightMatrix[(j*currLayer->numNeuronsCurrentLayer) + z]);
				}
				else {
					fprintf(layer_weights,"%lf\n", currLayer->weightMatrix[(j*currLayer->numNeuronsCurrentLayer) + z]);
				}
			}
		}
		// writing biases to the file
		for (int j = 0; j < currLayer->numNeuronsNextLayer; j++) {
			fprintf(layer_weights, "%lf\n", currLayer->biases[j]);
		}
		// closing the file since writing is done
		fclose(layer_weights);
	}

	
}

//function for training -> go through all images in training data and minimize mean squared error
void trainNeuralNet() {
	//
}

//function for evaluating (given a black and white picture, output the color image after evaluation of patches in neural net)
void evaluateNeuralNet(char* blackWhiteImageName) {

}
