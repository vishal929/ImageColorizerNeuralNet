#ifndef neuralkernal_h
#define neuralkernel_h
// function declarations for gpu kernels with neural net

void layerMultiplicationAddWrapper(double* weights, double* inputs, double* biases, double* output, int numNeuronsNextLayer, int numNeuronsCurrentLayer);



#endif 
