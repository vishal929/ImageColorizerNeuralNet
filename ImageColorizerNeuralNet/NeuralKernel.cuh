#ifndef neuralkernal_h
#define neuralkernel_h
// function declarations for gpu kernels with neural net

void layerMultiplicationWrapper(double* weights, double* inputs, double* biases, double* output, int numNeuronsNextLayer, int numNeuronsCurrentLayer);
void sigmoidWrapper(double* inputs, int numInputs);
void biasAddWrapper(double* results, double* biases, int numBiases);

#endif 
