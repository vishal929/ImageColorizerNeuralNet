// file serves as a container for logic associated with neural net specifics

#include <math.h>
#include "NeuralNet.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <CImg.h>
#include <random>
#include<fileapi.h>

// including cuda kernels
#include "NeuralKernel.cuh"
#include "imageKernel.cuh"

using namespace std;
using namespace cimg_library;

// initializing a neural net based on weights

// initializing a neural net based on random weights (before training)

// function to write weights back to txt file (for later loading/testing)
// each layer will have a different txt file with weights

// we may want to load a model from weights after training sessions so we do not "lose" progress
net* loadNeuralNet(int numInputsInData) {
	// we will check from "0weights.txt", "1weights.txt", etc. to find # of layers
	// if none found, we initialize neuralNet instead

	// important: if numInputsInData does not align with numNeurons in layer 0, then we should return NULL and print a warning to the user that the inputs do not match up!
	// (this indicates that they modified the patch size, but they want to retrain the same model! They should train a new model after deleting the weight txt files)
}

// initializes a neural net object with random small weight values -> output layer not included in count but input layer is
net* initializeNeuralNet(int numLayers, int numInputsInData) {
	net* toReturn = (net*)malloc(sizeof(net));
	toReturn->numLayers = numLayers;
	// allocating inner layers
	layer** innerLayers = (layer**)malloc(sizeof(layer*) * numLayers);
	for (int i = 0; i < numLayers; i++) {
		innerLayers[i] = (layer*) malloc(sizeof(layer));
		int neuronsNextLayer;
		if (i == numLayers - 1) {
			neuronsNextLayer = 3;
		}
		else {
			neuronsNextLayer = 100;
		}
		innerLayers[i]->numNeuronsNextLayer= neuronsNextLayer;
		if (i == 0) {	
			// input layer
			innerLayers[i]->numNeuronsCurrentLayer = numInputsInData;
		}
		else {
			innerLayers[i]->numNeuronsCurrentLayer = 100;
		}
		// initializing weight matrix to random small values
		srand(time(NULL));
		for (int j = 0; j < innerLayers[i]->numNeuronsCurrentLayer * innerLayers[i]->numNeuronsNextLayer; j++) {
			// random double between 0 and 0.25
			innerLayers[i]->weightMatrix[j] =double( rand()) / double((RAND_MAX * 4));
		}
		// initializing biases to zero
		for (int j = 0; j < innerLayers[j]->numNeuronsNextLayer; j++) {
			innerLayers[i]->biases[j] = 0;
		}
	}
	toReturn->numInputs = numInputsInData;
	toReturn->neuralLayers = innerLayers;

	return toReturn;
}

//we will want to periodically write back weights when training the model
void writeWeights(net* neuralNet) {
	for (int i = 0; i < neuralNet->numLayers; i++) {
		// writing the weights of each layer to a file in matrix form
		// the biases will be written after the matrix after a newline
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
		// writing biases to the file
		for (int j = 0; j < currLayer->numNeuronsNextLayer; j++) {
			fprintf(layer_weights, "%lf\n", currLayer->biases[j]);
		}
		// closing the file since writing is done
		fclose(layer_weights);
	}

	
}

//function for training -> go through all images in training data and minimize mean squared error
void trainNeuralNet() {
	
	int numTrainingImages = getNumTrainingImages();
	// idea is to choose a random picture in the training data folder
	// convert this image to black and white pixel values
	// then choose a random pixel
	// apply it to our neural net to predict RGB values
	// then do backpropogation
	// every 100 training sessions , we will write the weights out the txt file and then show total error of an image to the user
	int numTrainingSessions = 100;
	// we will use patch size 301x301 with the middle pixel being the pixel we wish to color
	int patchSize = 301;

	// loading the neural net or creating a new one if weights txt files are not found
	net* netToTrain = loadNeuralNet(patchSize * patchSize);

	// getting random number in a range
	random_device rando; 
	mt19937 gen(rando()); 
	uniform_int_distribution<> distr(0, numTrainingImages-1); // define the range
	int imageToPick = distr(gen);

	// choosing a random picture
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;

	hFind = FindFirstFile("./TestData", &FindFileData);
	
	for (int i = 0; i <= imageToPick; i++) {
		FindNextFile(hFind, &FindFileData);
	}

	// now the data is pointed to the picture to choose

	// using cimg to get data for the given picture
	CImg<int> colorPicture(FindFileData.cFileName);

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
		backPropogate(netOutputRGB, actualR, actualG, actualB, netToTrain);

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
	// returns the total error to the user

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

// we are doing a single sigmoid layer for now and we will get RGB output
double* evaluateNeuralNet(double* patch, net* netToRun) {
		
}

void backPropogate(double* outputRGB, int actualR, int actualG, int actualB, net* netToTrain) {

}

int getNumTrainingImages() {
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;
	int numPictures = 0;
	hFind = FindFirstFile("./TrainingData", &FindFileData);
	while (FindNextFile(hFind, &FindFileData) == true) {
		numPictures++;
	}
	FindClose(hFind);
	return numPictures;
}

int getNumTestImages() {
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;
	int numPictures = 0;
	hFind = FindFirstFile("./TestData", &FindFileData);
	while (FindNextFile(hFind, &FindFileData) == true) {
		numPictures++;
	}
	FindClose(hFind);
	return numPictures;
}
