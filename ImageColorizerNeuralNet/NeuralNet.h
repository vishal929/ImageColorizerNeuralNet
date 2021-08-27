#pragma once

// node represents the neurons in each layer of our neural net
typedef struct neuron {
	// neurons should have an input (this is the output of the previous layers neurons with respective weights)
	double input;
	// neurons should have a list of weights for each neuron in the next layer
	double* weights;
	// number of weights for iteration
	int numWeights;
} neuron;

typedef struct layer {
	// each layer should include neurons
	neuron** neurons;
	// number of neurons for iteration
	int numNeurons;
	// a layer should include a function that all neurons use
	double (*function)(double);
} layer;

typedef struct net {
	// a net should be a list of middle layers (in order) for propogation
	layer* neuralLayers;
	double* inputs;
	// including number of inputs for batch training
	int numInputs;
} net;
