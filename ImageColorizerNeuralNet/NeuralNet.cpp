// file serves as a container for logic associated with neural net specifics -> allows building and training the model on CPU
// The gpu specific logic will be implemented later -> the gpu can use the structs, but the function will need to be adjusted on the CUDA side

#include <math.h>
#include "NeuralNet.h"

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
