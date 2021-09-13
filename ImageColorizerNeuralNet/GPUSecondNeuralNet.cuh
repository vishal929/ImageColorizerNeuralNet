#ifndef GPUSecondNeuralNet_h 
#define GPUSecondNeuralNet_h

// input size is the size of the input layer (currently 100x100)
#define inputSize 10000
#define squareSide 100
// hidden layer numNeurons is the number of neurons in intermediate layers
#define hiddenLayerNumNeurons 8750
// output size is the number of neurons in the output layer (currently 50x50 with 3 RGB values for each)
#define outputSize 7500
#define outputSquareSide 50
// epochNumber is the number of training sessions to perform on a particular image
#define epochNum 20
// defining a standard number of layers to use (3 layers for now with 1 input layer and 1 hidden layer and 1 output layer)
#define standardNetSize 2

//defining a batch size for gpu evaluation speedup
#define numInputSquares 1

// defining the data organization of our net on the gpu
typedef struct GPUNet {
	// weights for our model
	double** weights;
	// weight adjustments for gradient descent backpropogation
	double** weightAdjustments;
	// delta values for backpropogation
	double** deltas;
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
void outputFromGPUNet(char* imageName, char* outputImageName);

#endif 