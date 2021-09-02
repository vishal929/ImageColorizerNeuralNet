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

// including cuda kernels
#include "NeuralKernel.cuh"
#include "imageKernel.cuh"

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

// we are doing a single sigmoid layer for now and we will get RGB output
double* evaluateNeuralNet(double* patch, net* netToRun) {
	// firstly setting the neural net inputs to the patch	
	netToRun->inputs = patch;
	double* output;
	for (int i = 0; i < netToRun->numLayers; i++) {
		layer* toConsider = netToRun->neuralLayers[i];
		if (i == 0) {
			// then we just copy the inputs to the layer inputs 
			memcpy(toConsider->neuronInputs, netToRun->inputs, sizeof(double) * netToRun->numInputs);
		}
		// matrix multiply of weights with inputs and adding biases
		output = (double*)malloc(sizeof(double) * toConsider->numNeuronsNextLayer);
		layerMultiplicationWrapper(toConsider->weightMatrix, toConsider->neuronInputs, toConsider->biases, output, toConsider->numNeuronsNextLayer, toConsider->numNeuronsCurrentLayer);

		// running sigmoid of the output 
		sigmoidWrapper(output, toConsider->numNeuronsNextLayer);

		// adding the bias vector at the end with cuda kernel
		// biasAddWrapper(output, toConsider->biases, toConsider->numNeuronsNextLayer);
		if (i != netToRun->numLayers - 1) {
			memcpy(netToRun->neuralLayers[i + 1]->neuronInputs, output, sizeof(double) * toConsider->numNeuronsNextLayer);
			free(output);
		}
	}
	//final 3 rgb values (before scaling by 255)
	return output;
}

// function for propogating backwards through the neural net and adjusting the weights/biases
void backPropogate(double* outputRGB, int actualR, int actualG, int actualB, net* netToTrain, double learningRate) {
	// if we need to implement relu beforehand, I will need to adjust this and the evaluate 
	// squared error of output (1/2 is for ease of taking derivatives with respect to inputs later)
	double error = 0.5 * (pow(outputRGB[0] - (actualR / 255), 2) + pow(outputRGB[1] - (actualG / 255), 2) + pow(outputRGB[2] - (actualB / 255), 2));
	// going backwards and adjusting the adjustweights matrix for each layer 
	// partial derivatives for error
	double dEdR = -(((double)(actualR / 255)) - outputRGB[0]);
	double dEdG = -(((double)(actualG / 255)) - outputRGB[1]);
	double dEdB = -(((double)(actualB / 255)) - outputRGB[2]);
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
		// maybe gpu implementation below if cpu too slow
		// void trainingHelperWrapper(net* toTrain, double* netOutput, double actualR, double actualG, double actualB, double learningRate);
	}
	// now that we have populated the weight adjustments, we can go through each layer and make the adjustments required, then we can free the adjustment memory
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
		}
		// initializing layer inputs
		innerLayers[i]->neuronInputs = (double*) malloc(sizeof(double) * neuronsCurrentLayer);
		// initializing layer biases
		innerLayers[i]->biases = (double*)malloc(sizeof(double) * neuronsNextLayer);
		// initially filling biases with 0
		for (int j = 0;j < neuronsNextLayer;j++) {
			innerLayers[i]->biases[j] = 0;
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
					fprintf(layer_weights,"%lf ", currLayer->weightMatrix[(j*currLayer->numNeuronsCurrentLayer) + z]);
				}
				else {
					fprintf(layer_weights,"%lf\n", currLayer->weightMatrix[(j*currLayer->numNeuronsCurrentLayer) + z]);
				}
			}
		}
		// writing biases
		for (int j = 0; j < currLayer->numNeuronsNextLayer; j++) {
			fprintf(layer_weights, "%lf\n", currLayer->biases[j]);
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
		int actualR = newR[(rowOfPixel * 2160) + colOfPixel];
		int actualG = newG[(rowOfPixel * 2160) + colOfPixel];
		int actualB = newB[(rowOfPixel * 2160) + colOfPixel];

		// going through backpropogation algorithm with the output
		backPropogate(netOutputRGB, actualR, actualG, actualB, netToTrain,learningRate);

		free(netOutputRGB);
		// freeing the input patch
		free(netToTrain->inputs);
		numEpochs++;
		if (numEpochs == numTrainingSessions) {
			// firstly writing the weights to the filesystem to "save" our training progress
			printf("DONE WRITING WEIGHTS TO FILESYSTEM!\n");
			// we want to see the complete error of this image, if it is very high, we want to continue training
			printf("ERROR FOR TRAINING IMAGE: %s: %lf\n", FindFileData.cFileName, 0.0);
		} 

	}
	printf("DONE TRAINING IMAGE: %s\n", FindFileData.cFileName);
	

}

// function for using the neural net on testing data
void testNeuralNet() {
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

//function for evaluating (given a black and white picture, output the color image after evaluation of patches in neural net)
void outputFromNeuralNet(char* blackWhiteImageName, char* colorOutputName) {
	// calling evaluate neural net for every patch and then using CIMG to save the output color image
}






