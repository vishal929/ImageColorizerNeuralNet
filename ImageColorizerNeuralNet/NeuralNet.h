#pragma once

typedef struct layer {
	// a layer should include a function that all neurons use (applied after biases are added to weighted average of inputs)
	//double (*function)(double);
	// each layer has inputs for every neuron
	double* neuronInputs;
	// each layer has a weight matrix for output for the next layer
	double* weightMatrix;
	// the number of rows in the weight matrix is the number of neurons in the next layer
	int numNeuronsNextLayer;
	// the number of columns in the weight matrix is the number of neurons in the current layer
	int numNeuronsCurrentLayer;
	// each layer has a set of biases for every neuron in the NEXT layer
	// TODO: A BIAS SHOULD BE A SINGLE EXTRA NEURON IN THE CURRENT LAYER WITH CONSTANT ACTIVATION=1 (so that the weight is the actual bias)
	double* biases;
} layer;

//NOTE: the last layer (output layer) will have 3 output neurons for R G and B respectively

typedef struct net {
	// a net should be a list of middle layers (in order) for propogation
	layer** neuralLayers;
	// a net should know the number of layers in itself
	int numLayers;
	// each neural net should have a list of inputs to propogate to the input layer
	double* inputs;
	// including number of inputs for batch training
	int numInputs;
} net;
