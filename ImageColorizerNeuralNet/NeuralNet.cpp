// file serves as a container for logic associated with neural net specifics

#include <math.h>
#include "NeuralNet.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <CImg.h>
#include <random>
#include<fileapi.h>
#include<tchar.h>
#include<process.h>

// including cuda kernels
#include "NeuralKernel.cuh"
#include "imageKernel.cuh"

#define MAX_THREADS 32


using namespace std;
using namespace cimg_library;

int getNumTrainingImages() {
	string searchName = "./TrainingData/*";
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;
	int numPictures = 0;
	hFind = FindFirstFile(searchName.c_str(), &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		cout << "OH NO TRAINING DATA NOT FOUND!\n";
		return 0;
	}
	else {
		numPictures++;
	}
	
	while (FindNextFile(hFind, &FindFileData)) {
		if ((FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
			numPictures++;
		}
	}

	FindClose(hFind);
	return numPictures;
}

int getNumTestImages() {
	string searchName = "./TestData/*";
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;
	int numPictures = 0;
	hFind = FindFirstFile(searchName.c_str(), &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		cout << "OH NO TEST DATA NOT FOUND!\n";
		return 0;
	}
	else {
		numPictures++;
	}
	
	while (FindNextFile(hFind, &FindFileData)) {
		if ((FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
			numPictures++;
		}
	}

	FindClose(hFind);
	return numPictures;
}

double sigmoid(double input) {
	return 1 / (1 + exp(-input));
}

// derivative calculation of sigmoid
double sigmoidDerivative(double value) {
	double sigmoided = sigmoid(value);
	return sigmoided * (1 - sigmoided);
}

void CPULayerMultiplicationAndAddition(double* outputBuffer, double* weights, double* inputs, double* biases, int nextLayerNumNeurons, int currentLayerNumNeurons) {
	// matrix vector multiplication
	for (int i = 0;i < nextLayerNumNeurons; i++) {
		for (int k = 0;k < currentLayerNumNeurons;k++) {
			outputBuffer[i] += weights[(i * currentLayerNumNeurons) + k] * inputs[k];
		}
		// adding biases to output
		outputBuffer[i] += biases[i];
	}


}

void cpuSigmoid(double* bufferToSigmoid, int numElements) {
	for (int i = 0;i < numElements;i++) {
		bufferToSigmoid[i] = sigmoid(bufferToSigmoid[i]);
	}
}

// we are doing a single sigmoid layer for now and we will get RGB output
double* evaluateNeuralNet(double* patch, net* netToRun) {
	// firstly setting the neural net inputs to the patch	
	netToRun->inputs = patch;
	double* output = NULL;
	for (int i = 0; i < netToRun->numLayers; i++) {
		layer* toConsider = netToRun->neuralLayers[i];
		if (i == 0) {
			// then we just copy the inputs to the layer inputs 
			memcpy(toConsider->neuronInputs, netToRun->inputs, sizeof(double) * netToRun->numInputs);
		}
		// matrix multiply of weights with inputs and adding biases
		output = (double*)malloc(sizeof(double) * toConsider->numNeuronsNextLayer);
		for (int i = 0;i < toConsider->numNeuronsNextLayer;i++) {
			output[i] = 0.0;
		}
		// layerMultiplicationWrapper(toConsider->weightMatrix, toConsider->neuronInputs, toConsider->biases, output, toConsider->numNeuronsNextLayer, toConsider->numNeuronsCurrentLayer);
		CPULayerMultiplicationAndAddition(output, toConsider->weightMatrix, toConsider->neuronInputs, toConsider->biases, toConsider->numNeuronsNextLayer, toConsider->numNeuronsCurrentLayer);
		if (isnan(output[0])) {
			cout << "ISSUE WITH OUTPUT BEFORE SIGMOID!\n";
		}

		// running sigmoid of the output 
		// sigmoidWrapper(output, toConsider->numNeuronsNextLayer);
		cpuSigmoid(output, toConsider->numNeuronsNextLayer);
		if (isnan(output[0])) {
			cout << "ISSUE WITH OUTPUT AFTER SIGMOID!\n";
		}

		// adding the bias vector at the end with cuda kernel
		// biasAddWrapper(output, toConsider->biases, toConsider->numNeuronsNextLayer);
		if (i != netToRun->numLayers - 1) {
			memcpy(netToRun->neuralLayers[i + 1]->neuronInputs, output, sizeof(double) * toConsider->numNeuronsNextLayer);
			free(output);
		}
	}
	//final 3 rgb values (before scaling by 255)
	if (output == NULL) {
		if (netToRun->numLayers == 0) {
			cout << "FOUND THE CULPRIT!\n";
		}
	}
	return output;
}

void backPropogate(double* outputRGB, int actualR, int actualG, int actualB, net* netToTrain, double learningRate) {
	// partial derivatives for error
	
	double dEdR = -((((double)actualR) / 255) - outputRGB[0]);
	double dEdG = -((((double)actualG) / 255) - outputRGB[1]);
	double dEdB = -((((double)actualB) / 255) - outputRGB[2]);
	double* currLayerOutput = (double*)malloc(sizeof(double) * 3);
	memcpy(currLayerOutput, outputRGB, sizeof(double) * 3);
	//double* nextDerivatives = (double*)malloc(sizeof(double) * 3);
	vector<double> nextDerivatives;
	nextDerivatives.push_back(dEdR * outputRGB[0] * (1-outputRGB[0]));
	nextDerivatives.push_back(dEdG * outputRGB[1] * (1-outputRGB[1]));
	nextDerivatives.push_back(dEdB * outputRGB[2] * (1-outputRGB[2]));
	
	for (int z = netToTrain->numLayers - 1; z >= 0;z--) {
		layer* toConsider = netToTrain->neuralLayers[z];
		toConsider->weightAdjustments = (double*)malloc(sizeof(double) * toConsider->numNeuronsCurrentLayer * toConsider->numNeuronsNextLayer);
		for (int i = 0;i < toConsider->numNeuronsNextLayer;i++) {
			for (int j = 0;j < toConsider->numNeuronsCurrentLayer;j++) {
				// setting adjustment to be completed later	
				toConsider->weightAdjustments[(i * toConsider->numNeuronsCurrentLayer) + j] = nextDerivatives[i] * toConsider->neuronInputs[j];
			}

			// we can update biases  immediately as they are not needed for any other error calculation
			toConsider->biases[i] -= learningRate * nextDerivatives[i];
		}
		//updating the derivatives vector for the next iteration
		vector<double> newDerivatives;
		for (int j = 0;j < toConsider->numNeuronsCurrentLayer;j++) {
			double derivativeSum = 0;
			for (int i = 0;i < toConsider->numNeuronsNextLayer;i++) {
				derivativeSum += nextDerivatives[i] * toConsider->weightMatrix[(i*toConsider->numNeuronsCurrentLayer) + j];
			}
			newDerivatives.push_back(derivativeSum * toConsider->neuronInputs[j] * (1 - toConsider->neuronInputs[j]));
		}
		// setting the old derivative vector to the new one
		nextDerivatives = newDerivatives;
	}

	// now doing the weight updates and freeing weight adjustment matrices
	for (int z = 0;z < netToTrain->numLayers;z++) {
		layer* toConsider = netToTrain->neuralLayers[z];
		for (int i = 0;i < toConsider->numNeuronsNextLayer;i++) {
			for (int j = 0;j < toConsider->numNeuronsCurrentLayer;j++) {
				toConsider->weightMatrix[(i * toConsider->numNeuronsCurrentLayer) + j] -= learningRate * toConsider->weightAdjustments[(i * toConsider->numNeuronsCurrentLayer) + j];
			}
		}
		free(toConsider->weightAdjustments);
	}

}

// function for propogating backwards through the neural net and adjusting the weights/biases
/*
void backPropogate(double* outputRGB, int actualR, int actualG, int actualB, net* netToTrain, double learningRate) {
	// if we need to implement relu beforehand, I will need to adjust this and the evaluate 
	// squared error of output (1/2 is for ease of taking derivatives with respect to inputs later)
	double error = 0.5 * (pow(outputRGB[0] - (((double)actualR) / 255), 2) + pow(outputRGB[1] - (((double)actualG) / 255), 2) + pow(outputRGB[2] - (((double)actualB) / 255), 2));
	// going backwards and adjusting the adjustweights matrix for each layer 
	// partial derivatives for error
	
	double dEdR = -((((double)actualR) / 255) - outputRGB[0]);
	double dEdG = -((((double)actualG) / 255) - outputRGB[1]);
	double dEdB = -((((double)actualB) / 255) - outputRGB[2]);
	double* currLayerOutput = (double*)malloc(sizeof(double) * 3);
	memcpy(currLayerOutput, outputRGB, sizeof(double) * 3);
	double* nextDerivatives = (double*)malloc(sizeof(double) * 3);
	nextDerivatives[0] = dEdR;
	nextDerivatives[1] = dEdG;
	nextDerivatives[2] = dEdB;
	for (int i = netToTrain->numLayers - 1; i >= 0; i--) {
		layer* toConsider = netToTrain->neuralLayers[i];
		toConsider->weightAdjustments = (double*)malloc(sizeof(double) * toConsider->numNeuronsCurrentLayer * toConsider->numNeuronsNextLayer);
		//backpropogating derivatives
		double* nextNextDerivatives = (double*)malloc(sizeof(double) * toConsider->numNeuronsCurrentLayer);
		// if 1 then we are on sigmoid output layer
		for (int j = 0; j < toConsider->numNeuronsNextLayer; j++) {
			for (int z = 0; z < toConsider->numNeuronsCurrentLayer; z++) {
				double adjustment = nextDerivatives[j] * currLayerOutput[j] * (1 - currLayerOutput[j]) * toConsider->neuronInputs[z];
				
				toConsider->weightAdjustments[(j * toConsider->numNeuronsCurrentLayer) + z] = adjustment;
				nextNextDerivatives[z] += nextDerivatives[j] * currLayerOutput[j] * (1 - currLayerOutput[j]) * toConsider->weightMatrix[(j * toConsider->numNeuronsCurrentLayer) + z];
			}
			//adjusting bias term directly (not needed for any future updates)
			toConsider->biases[j] -= learningRate * nextDerivatives[j] * currLayerOutput[j] * (1 - currLayerOutput[j]);
		}

		if (i == 1) {
			free(currLayerOutput);
		}

		free(nextDerivatives);
		currLayerOutput = toConsider->neuronInputs;
		nextDerivatives = nextNextDerivatives;
		
	}
	// now that we have populated the weight adjustments, we can go through each layer and make the adjustments required, then we can free the adjustment memory
	for (int i = 0;i < netToTrain->numLayers;i++) {
		layer* toConsider = netToTrain->neuralLayers[i];
		for (int j = 0;j < toConsider->numNeuronsNextLayer;j++) {
			for (int z = 0;z < toConsider->numNeuronsCurrentLayer;z++) {
				toConsider->weightMatrix[(j*toConsider->numNeuronsCurrentLayer) + z] -= learningRate * toConsider->weightAdjustments[(j*toConsider->numNeuronsCurrentLayer) + z];
			}
		}
		free(toConsider->weightAdjustments);
	}
	// maybe gpu implementation below if cpu too slow
	// trainingHelperWrapper(netToTrain,  outputRGB, actualR, actualG, actualB, learningRate);
} */

// goes through every patch in the image, gets the error and adjusts the buffer accordingly
void testSpecificNeuralNet(net* netToTrain, double* RGBErrorBuffer, const char* pictureToTest) {
	const int patchSize = 301;
	CImg<int> colorPicture(pictureToTest);

	// calling CUDA kernel to get the black and white image
	int* colorR = colorPicture.data();
	int* colorG = colorR + (colorPicture.height() * colorPicture.width());
	int* colorB = colorG + (colorPicture.height() * colorPicture.width());
	int* bwBuffer = (int*)malloc(sizeof(int) * colorPicture.width() * colorPicture.height());

	makeImageBlackAndWhiteWrapper(colorR, colorG, colorB, bwBuffer, colorPicture.height(), colorPicture.width());

	/* Converting picture to 4k black and white and then picking a random pixel*/

	int* newBWValues = (int*)malloc(sizeof(int) * 3840 * 2160);
	// calling kernel wrapper for gpu function
	makeBlackWhiteImage4KWrapper(bwBuffer, newBWValues, colorPicture.height(), colorPicture.width());
	free(bwBuffer);
	// need to also convert color image to 4k for comparison and backpropogation
	int* newR = (int*)malloc(sizeof(int) * 3840 * 2160);
	int* newG = (int*)malloc(sizeof(int) * 3840 * 2160);
	int* newB = (int*)malloc(sizeof(int) * 3840 * 2160);
	makeColorImage4kWrapper(colorR, colorG, colorB, newR, newG, newB, colorPicture.height(), colorPicture.width());

	/*Scaling greyscale values to be between 0 and 1*/
	double* scaledBWValues = (double*)malloc(sizeof(double) * 3840 * 2160);
	pixelScaleWrapper(newBWValues, scaledBWValues, 3840, 2160);
	free(newBWValues);

	/*Going through random 3000 single pixel patch and getting the error only for the original input*/
	int testedCount = 0;
	while (testedCount != 3000) {
		testedCount++;
		random_device rando;
		mt19937 gen(rando());
		uniform_int_distribution<> row(0, colorPicture.height()-1);
		uniform_int_distribution<> col(0, colorPicture.width()-1); 
		int rowOfPixel= row(gen);
		int colOfPixel= col(gen);
		double* imagePatch = (double*)malloc(sizeof(double) * patchSize * patchSize);
		getPatchWrapper(scaledBWValues, imagePatch, 3840, 2160, patchSize, 1, rowOfPixel, colOfPixel);
		// getting the output of this patch
		double* netOutputRGB = evaluateNeuralNet(imagePatch, netToTrain);
		int actualR = newR[(rowOfPixel* 2160) + colOfPixel];
		int actualG = newG[(rowOfPixel* 2160) + colOfPixel];
		int actualB = newB[(rowOfPixel* 2160) + colOfPixel];
		// adjusting error in buffers
		RGBErrorBuffer[0] += pow(netOutputRGB[0] - (((double)actualR) / 255.0), 2);
		RGBErrorBuffer[1] += pow(netOutputRGB[1] - ((double) actualG / 255.0), 2);
		RGBErrorBuffer[2] += pow(netOutputRGB[2] - ((double) actualB / 255.0), 2);
		cout << "on test number: " << testedCount << " out of 3000 with total error: " << (0.5) * (RGBErrorBuffer[0] + RGBErrorBuffer[1] + RGBErrorBuffer[2]) << "\n";
		//freeing memory for this patch
		free(imagePatch);
		free(netOutputRGB);
	}
	/*
	for (int i = 0;i < colorPicture.height();i++) {
		for (int j = 0;j < colorPicture.width();j++) {
			cout << "on row: " << i << " and col: " << j << "\n";
			// getting the patch for this pixel
			double* imagePatch = (double*)malloc(sizeof(double) * patchSize * patchSize);
			getPatchWrapper(scaledBWValues, imagePatch, 3840, 2160, patchSize, 1, i, j);
			// getting the output of this patch
			double* netOutputRGB = evaluateNeuralNet(imagePatch, netToTrain);
			int actualR = newR[(i* 2160) + j];
			int actualG = newG[(i* 2160) + j];
			int actualB = newB[(i* 2160) + j];
			// adjusting error in buffers
			RGBErrorBuffer[0] += pow(netOutputRGB[0] - (actualR / 255), 2);
			RGBErrorBuffer[1] += pow(netOutputRGB[1] - (actualG / 255), 2);
			RGBErrorBuffer[2] += pow(netOutputRGB[2] - (actualB / 255), 2);
			//freeing memory for this patch
			free(imagePatch);
			free(netOutputRGB);
		}
	} */

	cout << "RED ERROR FOR IMAGE: " << pictureToTest << " ERROR: " << RGBErrorBuffer[0] << "\n";
	cout << "GREEN ERROR FOR IMAGE: " << pictureToTest << " ERROR: " << RGBErrorBuffer[1] << "\n";
	cout << "BLUE ERROR FOR IMAGE: " << pictureToTest << " ERROR: " << RGBErrorBuffer[2] << "\n";
}

// function to allocate memory for another net and copy data to it
net* copyNet(net* toCopy) {
	net* copied = (net*)malloc(sizeof(net));
	copied->numInputs = toCopy->numInputs;
	copied->numLayers = toCopy->numLayers;
	copied->neuralLayers = (layer**)malloc(sizeof(layer*) * copied->numLayers);
	for (int i = 0;i < copied->numLayers;i++) {
		layer* toAllocate = (layer*)malloc(sizeof(layer));
		toAllocate->numNeuronsCurrentLayer = toCopy->neuralLayers[i]->numNeuronsCurrentLayer;
		toAllocate->numNeuronsNextLayer= toCopy->neuralLayers[i]->numNeuronsNextLayer;
		toAllocate->neuronInputs = (double*)malloc(sizeof(double) * toAllocate->numNeuronsCurrentLayer);
		toAllocate->weightMatrix = (double*)malloc(sizeof(double) * toAllocate->numNeuronsCurrentLayer * toAllocate->numNeuronsNextLayer);
		toAllocate->biases = (double*)malloc(sizeof(double) * toAllocate->numNeuronsNextLayer);

		//copying weights
		memcpy(toAllocate->weightMatrix, toCopy->neuralLayers[i]->weightMatrix, sizeof(double) * toAllocate->numNeuronsCurrentLayer * toAllocate->numNeuronsNextLayer);
		// copying biases
		memcpy(toAllocate->biases, toCopy->neuralLayers[i]->biases, sizeof(double) * toAllocate->numNeuronsNextLayer);
		copied->neuralLayers[i] = toAllocate;
	}
	//returning the deep copied net struct
	return copied;
}

// function to free a net
void freeNet(net* toFree) {
	for (int i = 0;i < toFree->numLayers;i++) {
		layer* inQuestion = toFree->neuralLayers[i];
		free(inQuestion->biases);
		free(inQuestion->neuronInputs);
		free(inQuestion->weightMatrix);
		free(inQuestion);
	}
	free(toFree->neuralLayers);
	free(toFree);
}

// we may want to load a model from weights after training sessions so we do not "lose" progress
net* loadNeuralNet(int numLayers, int numInputsInData) {
	// we will check from "0weights.txt", "1weights.txt", etc. to find # of layers
	// if none found, we initialize neuralNet instead

	// important: if numInputsInData does not align with numNeurons in layer 0, then we should return NULL and print a warning to the user that the inputs do not match up!
	// (this indicates that they modified the patch size, but they want to retrain the same model! They should train a new model after deleting the weight txt files)

	// I will probably not have more than 5 layers in the neural net for this problem
	net* toReturn = (net*)malloc(sizeof(net));
	toReturn->neuralLayers = (layer**)malloc(sizeof(layer*) * numLayers);
	toReturn->numLayers = numLayers;
	toReturn->numInputs = numInputsInData;
	for (int i = 0; i < numLayers; i++) {
		char* weightFileName = (char*)malloc(sizeof(char) * 13);
		weightFileName[0] = '0' + i;
		memcpy(weightFileName + 1, &"weights.txt\0", sizeof(char) * 12);
		FILE* weightsFile = fopen(weightFileName, "r");
		if (weightsFile == NULL && i == 0) {
			// then weights dont exist
			free(toReturn->neuralLayers);
			free(toReturn);
			free(weightFileName);
			return NULL;
		}
		else if (weightsFile == NULL && i != 0) {
			// incomplete net
			for (int j = 0; j < i; j++) {
				free(toReturn->neuralLayers[j]->biases);
				free(toReturn->neuralLayers[j]->weightMatrix);
				free(toReturn->neuralLayers[j]->neuronInputs);
				free(toReturn->neuralLayers[j]);
			}
			free(toReturn->neuralLayers);
			free(toReturn);
			free(weightFileName);
			return NULL;
		}
		else {
			// then extracting data into the layer struct
			toReturn->neuralLayers[i] = (layer*)malloc(sizeof(layer));
			int numNeurons, numNeuronsNextLayer;
			fscanf(weightsFile, "%d %d\n", &numNeurons, &numNeuronsNextLayer);
			toReturn->neuralLayers[i]->numNeuronsCurrentLayer = numNeurons;
			toReturn->neuralLayers[i]->numNeuronsNextLayer = numNeuronsNextLayer;
			toReturn->neuralLayers[i]->biases = (double*) malloc(sizeof(double) * numNeuronsNextLayer);
			toReturn->neuralLayers[i]->weightMatrix = (double*)malloc(sizeof(double) * numNeurons * numNeuronsNextLayer);
			toReturn->neuralLayers[i]->neuronInputs = (double*)malloc(sizeof(double) * numNeurons);
			// getting matrix data
			for (int j = 0; j < numNeuronsNextLayer; j++) {
				for (int z = 0; z < numNeurons; z++) {
					if (z == numNeurons - 1) {
						fscanf(weightsFile, "%lf\n", &(toReturn->neuralLayers[i]->weightMatrix[(j * numNeurons) + z]));
					}
					else {
						fscanf(weightsFile, "%lf ", &(toReturn->neuralLayers[i]->weightMatrix[(j * numNeurons) + z]));
					}
				}
			}
			// getting biases
			for (int j = 0; j < numNeuronsNextLayer; j++) {
				fscanf(weightsFile, "%lf\n", &(toReturn->neuralLayers[i]->biases[j]));
			}
		}
		free(weightFileName);
		fclose(weightsFile);
	}
	cout << "LOADED EXISTING NEURAL NET SUCCESSFULLY!\n";
	return toReturn;

}

// initializes a neural net object with random small weight values -> output layer not included in count but input layer is
net* initializeNeuralNet(int numLayers, int numInputsInData) {
	net* toReturn = (net*)malloc(sizeof(net));
	toReturn->numLayers = numLayers;
	toReturn->numInputs = numInputsInData;
	//toReturn->inputs = (double*)malloc(sizeof(double) * numInputsInData);
	// allocating inner layers
	layer** innerLayers = (layer**)malloc(sizeof(layer*) * numLayers);
	for (int i = 0; i < numLayers; i++) {
		innerLayers[i] = (layer*) malloc(sizeof(layer));
		int neuronsNextLayer, neuronsCurrentLayer;
		if (i == numLayers - 1) {
			neuronsNextLayer = 3;
		}
		else {
			neuronsNextLayer = 100;
		}
		innerLayers[i]->numNeuronsNextLayer= neuronsNextLayer;
		if (i == 0) {	
			// input layer
			neuronsCurrentLayer = numInputsInData;
			innerLayers[i]->numNeuronsCurrentLayer = numInputsInData;
		}
		else {
			neuronsCurrentLayer = 100;
			innerLayers[i]->numNeuronsCurrentLayer = 100;
		}
		// allocating weight matrix
		innerLayers[i]->weightMatrix = (double*) malloc(sizeof(double) * neuronsNextLayer * neuronsCurrentLayer);
		// initializing weight matrix to random small values
		srand(time(NULL));
		for (int j = 0; j < innerLayers[i]->numNeuronsCurrentLayer * innerLayers[i]->numNeuronsNextLayer; j++) {
			// random double between 0 and 0.25
			innerLayers[i]->weightMatrix[j] =double( rand()) / double((RAND_MAX * 4));
			// trying larger weights
			// innerLayers[i]->weightMatrix[j] = 100;
		}
		// initializing layer inputs
		innerLayers[i]->neuronInputs = (double*) malloc(sizeof(double) * neuronsCurrentLayer);
		// initializing layer biases
		innerLayers[i]->biases = (double*)malloc(sizeof(double) * neuronsNextLayer);
		// initially filling biases with small values
		for (int j = 0;j < neuronsNextLayer;j++) {
			innerLayers[i]->biases[j] = double( rand()) / double((RAND_MAX * 4));
		}
	}
	toReturn->neuralLayers = innerLayers;

	return toReturn;
}

//we will want to periodically write back weights when training the model
void writeWeights(net* neuralNet) {
	for (int i = 0; i < neuralNet->numLayers; i++) {
		// writing the weights of each layer to a file in matrix form
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
					fprintf(layer_weights,"%.3lf ", currLayer->weightMatrix[(j*currLayer->numNeuronsCurrentLayer) + z]);
				}
				else {
					fprintf(layer_weights,"%.3lf\n", currLayer->weightMatrix[(j*currLayer->numNeuronsCurrentLayer) + z]);
				}
			}
		}
		// writing biases
		for (int j = 0; j < currLayer->numNeuronsNextLayer; j++) {
			fprintf(layer_weights, "%.3lf\n", currLayer->biases[j]);
		}
		// closing the file since writing is done
		fclose(layer_weights);
		// freeing memory
		free(basename);
	}

	
}

//function for training -> go through all images in training data and minimize mean squared error
void trainNeuralNet(int numTrainingSessions, double learningRate) {
	// we will use patch size 301x301 with the middle pixel being the pixel we wish to color
	int patchSize = 301;
	// we will use 2 layers for now (input with sigmoid hidden layer into output)
	net* netToTrain= loadNeuralNet(2, patchSize * patchSize);
	if (netToTrain== NULL) {
		netToTrain = initializeNeuralNet(2, patchSize * patchSize);
	}
	int numTrainingImages = getNumTrainingImages();
	if (numTrainingImages == 0) {
		cout << "OH NO!!!!! We have an empty training set! Training aborted\n";
		return;
	}
	// idea is to choose a random picture in the training data folder
	// convert this image to black and white pixel values
	// then choose a random pixel
	// apply it to our neural net to predict RGB values
	// then do backpropogation
	
	// getting random number in a range
	random_device rando; 
	mt19937 gen(rando()); 
	uniform_int_distribution<> distr(1, numTrainingImages); 
	int imageToPick = distr(gen);

	// choosing a random picture
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind=NULL;

	int currCount = 0;
	string dirName = "TrainingData/";
	while (currCount != imageToPick) {
		if (hFind == NULL) {
			hFind = FindFirstFile("./TrainingData/*\0", &FindFileData);
		}
		else {
			FindNextFile(hFind, &FindFileData);
		}
		if ((FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
			currCount++;
		}
	}
	// now the data is pointed to the picture to choose

	// using cimg to get data for the given picture
	for (int i = 0;i < strlen(FindFileData.cFileName);i++) {
		dirName.push_back(FindFileData.cFileName[i]);
	}
	CImg<int> colorPicture(dirName.c_str());

	// calling CUDA kernel to get the black and white image
	int* colorR = colorPicture.data();
	int* colorG = colorR + (colorPicture.height() * colorPicture.width());
	int* colorB = colorG + (colorPicture.height() * colorPicture.width());
	int* bwBuffer = (int*)malloc(sizeof(int) * colorPicture.width() * colorPicture.height());

	makeImageBlackAndWhiteWrapper(colorR, colorG, colorB, bwBuffer, colorPicture.height(), colorPicture.width());
	
	/* Converting picture to 4k black and white and then picking a random pixel*/

	int* newBWValues = (int*)malloc(sizeof(int) * 3840 * 2160);
	// calling kernel wrapper for gpu function
	makeBlackWhiteImage4KWrapper(bwBuffer, newBWValues, colorPicture.height(), colorPicture.width());
	free(bwBuffer);
	// need to also convert color image to 4k for comparison and backpropogation
	int* newR = (int*)malloc(sizeof(int) * 3840 * 2160);
	int* newG = (int*)malloc(sizeof(int) * 3840 * 2160);
	int* newB = (int*)malloc(sizeof(int) * 3840 * 2160);
	makeColorImage4kWrapper(colorR, colorG,colorB,  newR,  newG,  newB, colorPicture.height(), colorPicture.width());

	/*Scaling greyscale values to be between 0 and 1*/
	double* scaledBWValues = (double*)malloc(sizeof(double) * 3840 * 2160);
	pixelScaleWrapper(newBWValues, scaledBWValues, 3840, 2160);
	free(newBWValues);
	// now scaled BWValues holds the scaled image pixels
		
	
	int numEpochs = 0;
	while (numEpochs != numTrainingSessions) {
		cout << "on epoch number: " << numEpochs << "\n";
		/* Picking a random patch to run through our neural net*/
		random_device rando;
		mt19937 gen(rando());
		uniform_int_distribution<> row(0, 3839);
		uniform_int_distribution<> col(0, 2159); 
		int rowOfPixel= row(gen);
		int colOfPixel= col(gen);
		double* imagePatch = (double*) malloc(sizeof(double) * patchSize * patchSize);
		getPatchWrapper(scaledBWValues,imagePatch, 3840, 2160, patchSize, 1, rowOfPixel, colOfPixel);

		

		// running the patch through our neural net
		double* netOutputRGB = evaluateNeuralNet(imagePatch, netToTrain);
		if (isnan(netOutputRGB[0])) {
			cout << "issue with output of evaluateNeuralNet!\n";
		}
		int actualR = newR[(rowOfPixel * 2160) + colOfPixel];
		int actualG = newG[(rowOfPixel * 2160) + colOfPixel];
		int actualB = newB[(rowOfPixel * 2160) + colOfPixel];

		// going through backpropogation algorithm with the output
		// BACKPROP RESULTING IN NAN VALUES INVESTIGATE WHY
		backPropogate(netOutputRGB, actualR, actualG, actualB, netToTrain,learningRate);

		free(netOutputRGB);
		// freeing the input patch
		free(imagePatch);
		numEpochs++;
		if (numEpochs == numTrainingSessions) {
			// firstly writing the weights to the filesystem to "save" our training progress
			writeWeights(netToTrain);
			printf("DONE WRITING WEIGHTS TO FILESYSTEM!\n");
			// we want to see the complete error of this image, if it is very high, we want to continue training
			// 3 values for red error, green error, and blue error respectively
			double completeError[3] = { 0,0,0 };
			testSpecificNeuralNet(netToTrain, completeError, dirName.c_str());
		} 

	}
	printf("DONE TRAINING IMAGE: %s\n", FindFileData.cFileName);
	

}

// function for using the neural net on testing data
// adjusts the RGB buffer to contain the specific rgb errors
void testNeuralNet(net* netToTrain, double* RGBErrorBuffer) {
	int numTestImages = getNumTestImages();
	// chooses a random image in the test data folder
	// converts to black and white
	// runs every single patch through the neural net and evaluates mean squared error for each RGB value
	// returns the total squared error to the user

	// getting random number in a range
	random_device rando; 
	mt19937 gen(rando()); 
	uniform_int_distribution<> distr(0, numTestImages-1); // define the range
	int imageToPick = distr(gen);

	// choosing a random picture
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;

	hFind = FindFirstFile("./TestData", &FindFileData);
	
	for (int i = 0; i <= imageToPick; i++) {
		FindNextFile(hFind, &FindFileData);
	}
	// getting error for this image
	printf("ERROR FOR TESTING NET ON IMAGE %s: %lf\n", FindFileData.cFileName, 0.0);
}

typedef struct multiThreadEvaluationArgs {
	double* scaledPixels;
	double* finalOutput;
	int row;
	int col;
	int rowDim;
	int colDim;
	net* toRun;
} multiThreadEvaluationArgs;

// function for a pthread to execute for multiple setting
unsigned int WINAPI getPatchEvaluateNetSetColor(void* netArgs) {
	int patchSize = 301;
	multiThreadEvaluationArgs* args = (multiThreadEvaluationArgs*)netArgs;
	// arguments
	double* scaledPixels = args->scaledPixels;
	double* finalOutput = args->finalOutput;
	// creating a copy of the neural net to run
	net* toRun = copyNet(args->toRun);
	//row and column of pixel to get patch for
	int row= args->row;
	int col= args->col;
	int rowDim = args->rowDim;
	int colDim = args->colDim;
	// this thread will grab the patch
	double* patch = (double*) malloc(sizeof(double) * patchSize * patchSize);
	getPatchWrapper(scaledPixels, patch, rowDim, colDim, patchSize, 1, row, col);
	// evaluating the patch
	double* rgb = evaluateNeuralNet(patch, toRun);

	// setting the results to the finalOutput buffer
	for (int i = 1;i <= 3;i++) {
		finalOutput[(row * colDim * i) + col] = rgb[i] * 255;
	}

	cout << "FINISHED SETTING FOR ROW: " << row << " and COL: " << col << "\n";

	// freeing memory
	free(args);
	freeNet(toRun);
	free(patch);
	free(rgb);

	return 0;
}

//function for evaluating (given a black and white picture, output the color image after evaluation of patches in neural net)
void outputFromNeuralNet(char* blackWhiteImageName, char* colorOutputName) {
	int patchSize = 301;
	// calling evaluate neural net for every patch and then using CIMG to save the output color image ** ASSUMES A NEURAL NET IS ALREADY TRAINED**
	net* toUse = loadNeuralNet(2, patchSize * patchSize);
	
	//grabbing the blackWhiteImage pixel data
	CImg<int> blackWhite(blackWhiteImageName);

	// allocating memory to modify
	double* scaledPixels = (double*)malloc(sizeof(double) * blackWhite.width() * blackWhite.height());

	// allocating final buffer ( times 3 for RGB values)
	double* finalOutput = (double*)malloc(sizeof(double) * 3 * blackWhite.width() * blackWhite.height());

	// gpu scaling image to apply to net
	pixelScaleWrapper(blackWhite.data(), scaledPixels, blackWhite.height(), blackWhite.width());

	// going through pixels and applying net matrix multiplication/addition of biases to each rgb value	

	// setting up thread args to copy from
	multiThreadEvaluationArgs* baseStruct = (multiThreadEvaluationArgs*) malloc(sizeof(multiThreadEvaluationArgs));
	baseStruct->finalOutput = finalOutput;
	baseStruct->scaledPixels = scaledPixels;
	baseStruct->colDim = blackWhite.width();
	baseStruct->rowDim = blackWhite.height();
	baseStruct->toRun = toUse;

	//keeping track of threads -> not to exceed 16 concurrent kernel calls
	HANDLE launchedThreads[MAX_THREADS];
	int numThreads = 0;
	for (int i = 0;i < blackWhite.height();i++) {
		for (int j = 0;j < blackWhite.width();j++) {
			// setting up the struct to pass to the thread
			multiThreadEvaluationArgs* threadArgs = (multiThreadEvaluationArgs*)malloc(sizeof(multiThreadEvaluationArgs));
			memcpy(threadArgs, baseStruct, sizeof(multiThreadEvaluationArgs));
			threadArgs->col = j;
			threadArgs->row = i;
			// launching thread
			//uintptr_t createdThread = _beginthread(getPatchEvaluateNetSetColor, 0, threadArgs);
			HANDLE threadHandle = (HANDLE)_beginthreadex(0, 0, getPatchEvaluateNetSetColor, threadArgs, 0, 0);
			// adding the handle to our list of handles
			launchedThreads[numThreads] = threadHandle;
			numThreads++;
			if (numThreads == MAX_THREADS) {
				//waiting on the current threads to finish execution before launching any more
				for (int z = 0;z < MAX_THREADS;z++) {
					if (launchedThreads[z] != NULL) {
						WaitForSingleObject(launchedThreads[z], INFINITE);
						CloseHandle(launchedThreads[z]);
						launchedThreads[z] = NULL;
						numThreads--;
					}
				}
			}
			/*
			// getting the patch 
			double* grabbedPatch = (double*)malloc(sizeof(double) * patchSize * patchSize);
			getPatchWrapper(scaledPixels, grabbedPatch, blackWhite.height(), blackWhite.width(), patchSize, 1, i, j);

			// getting rgb output
			double* rgbOutput = evaluateNeuralNet(grabbedPatch, toUse);

			// setting rgb output for result
			for (int z = 1;z <= 3;z++) {
				finalOutput[(i * blackWhite.width() * z) + j] = rgbOutput[z] * 255;
			}

			//freeing memory
			free(grabbedPatch);
			free(rgbOutput);
			cout << "setup row: " << i << "and column: " << j << "\n";
			*/
		}
	}
	free(baseStruct);

	// creating the final image
	CImg<double> finalImage(finalOutput, blackWhite.width(), blackWhite.height(), 1, 3);
	finalImage.save(colorOutputName);
	
}






