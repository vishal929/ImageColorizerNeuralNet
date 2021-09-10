#ifndef GPUSecondNeuralNet_h 
#define GPUSecondNeuralNet_h

// input size is the size of the input layer (currently 500x500)
#define inputSize 250000
#define squareSide 500
// hidden layer numNeurons is the number of neurons in intermediate layers
#define hiddenLayerNumNeurons 300
// output size is the number of neurons in the output layer (currently 250x250 with 3 RGB values for each)
#define outputSize 187500
// epochNumber is the number of training sessions to perform on a particular image
#define epochNum 300
// defining a standard number of layers to use (2 layers for now with 1 input layer and 1 hidden layer)
#define standardNetSize 2

// defining the data organization of our net on the gpu
typedef struct GPUNet {
	// weights for our model
	double** weights;
	// weight adjustments for gradient descent backpropogation
	double** weightAdjustments;
	// biases for our model
	double** biases;
	// adjustment for biases
	//double** biasAdjustments;
	// inputs for each layer
	double** layerInput;
	// number of inputs for each layer
	int* numInputs;
	// number of outputs for each layer
	int* numOutputs;
	// number of layers in our model
	int numLayers;
} GPUNet;

/*FUNCTION DECLARATIONS -> USER ONLY NEED TO KNOW TRAIN, OUTPUT , AND TEST*/
void writeGPUNet(GPUNet* net);
GPUNet* loadGPUNet();
void trainFromDataSet(double learningRate);
void sigmoidMatrixTest(double* input, double* output, int dim);

#endif 