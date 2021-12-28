#ifndef GPUSecondNeuralNet_h 
#define GPUSecondNeuralNet_h

// input size is the size of the input layer (currently 31x31)
#define inputSize 961
#define squareSide 31
// hidden layer numNeurons is the number of neurons in intermediate layers
#define hiddenLayerNumNeurons 1500
// output size is the number of neurons in the output layer (currently 1 x 1 with 3 values for r,g,b)
#define outputSize 3
#define outputSquareSide 1
// epochNumber is the number of training sessions to perform on random image patches until we start testing, this is multiplied with the batch size to get the number of samples!
#define epochNum 50
// defining a standard number of layers to use (3 layers for now with 1 input layer and 1 hidden layer and 1 output layer)
#define standardNetSize 3

//defining a batch size for gpu evaluation speedup (we evaluate 150000 patches at the same time and average)
#define numInputSquares 200000

//defining a batch size for training (this is a different batch size from evaluating since during training we access the memory of multiple images, incurring slowdown)
#define trainingBatch 1000

// defining a scalar for input pixels (we will scale by 128 for lab)
#define inputPixelScaler 255


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
double testFromTestData(GPUNet* toTest);
boolean testCublas();

#endif 