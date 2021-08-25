// file serves as a container for logic associated with neural net specifics

#include <math.h>

double inputFunction(double input);

typedef struct layer {
	// a layer should include a function
	double (*function)(double);
	// each layer should have some list of weights associated with it to form a matrix for the next layer
	double* weights;
	// row for weights matrix
	int weightsRows;
	// columns for weights matrix;
	int weightsCols;
} layer;

typedef struct net {
	// a net should be a list of middle layers (in order) for propogation
	layer* neuralLayers;
	double* inputs;
	// including number of inputs for batch training
	int numInputs;
} net;

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
