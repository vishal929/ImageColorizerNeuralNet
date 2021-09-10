// purpose of this file is to house GPU kernels associated with training and evaluating the neural net
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaErrorHandler.cuh"
#include "cublas.h"
#include <vector>
#include <Cimg.h>
#include <string>
#include <iostream>
#include <random>
#include <curand.h>
#include "GPUSecondNeuralNet.cuh"
#include "ImageKernel.cuh"



using namespace std;
using namespace cimg_library;



// my idea here is to create a more memory hitting neural net in order to flex the gpu more and speedup training/evaluating and output times
// idea is to take a 1000 x 1000 portion of the image, and guess the middle 500x500 pixel values so the output will be of size 500 x 500 x 3 for RGB values
	// this way, we can extend smaller images with black values to apply to the model and we can batch portions of larger images more easily
	// I will test training and if I cannot get a fit (which is likely with only 100 neurons a layer, I will try increasing the number of neurons)
	// if there is still sufficient gpu memory not being utilized, I can increase the input size and the output size and the number of neurons per layer to get a good mix

CImg<int> getRandomTrainingImage() {
	string searchName = "./TrainingData/*";
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;
	int numPictures = 0;
	hFind = FindFirstFile(searchName.c_str(), &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		cout << "OH NO TRAINING DATA NOT FOUND!\n";
		//returning empty image
		FindClose(hFind);
		return CImg<int>();
	}
	else {
		numPictures++;
	}

	while (FindNextFile(hFind, &FindFileData)) {
		if ((FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
			numPictures++;
		}
	}

	if (numPictures == 0) {
		cout << "OH NO!!!!! We have an empty training set! Training aborted\n";
		//returning empty image
		FindClose(hFind);
		return CImg<int>();
	}
	// idea is to choose a random picture in the training data folder
	// getting random number in a range
	random_device rando;
	mt19937 gen(rando());
	uniform_int_distribution<> distr(1, numPictures);
	int imageToPick = distr(gen);

	// choosing a random picture

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
	cout << "grabbed training image: " << dirName << "\n";
	FindClose(hFind);
	return colorPicture;
}

__device__ double sigmoidInput(double input) {
	return 1 / (1 + exp(-input));
}

__global__ void sigmoidMatrix(double* output, int dim) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i = tidx;i < dim;i += gridDim.x * blockDim.x) {
		output[i] = sigmoidInput(output[i]);
	}
}

//asserting that sigmoid matrix works (we will verify gpu output with cpu)
void sigmoidMatrixTest(double* input, double* output, int dim) {
	//creating copy of input to run sigmoidMatrix on
	double* deviceOutput;
	cudaErrorCheck(cudaMalloc(&deviceOutput, sizeof(double) * dim));
	cudaErrorCheck(cudaMemcpy(deviceOutput, input, sizeof(double) * dim, cudaMemcpyHostToDevice));
	sigmoidMatrix << <20, 512 >> > (deviceOutput, dim);
	cudaErrorCheck(cudaMemcpy(output, deviceOutput, sizeof(double) * dim, cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaFree(deviceOutput));
	//asserting that kernel ran successfully
	for (int i = 0;i < dim;i++) {
		double cpuSigmoid = 1 / (1 + exp(-input[i]));
		if ( cpuSigmoid != output[i]) {
			cout << "ERROR GPU RETURNED: " << output[i] << "BUT CPU RETURNED: " << cpuSigmoid << "\n";
			return;
		}
		if (isnan(output[i])) {
			cout << "NAN ERROR on INDEX: " << i << "\n";
			return;
		}
	}
	cout << "SIGMOID KERNEL WORKS!\n";
}

void cpuSigmoidMatrix(double* output, int dim) {
	for (int i = 0;i < dim;i++) {
		output[i] = 1 / (1 + exp(-output[i]));
	}
}

// given weight matrices and biases , we can initialize weights with random small numbers and biases with 0
__global__ void initializeGPUNet(double* weights, double* randomNumbers, double* biases, int numInputs, int numOutputs) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	for (int j = tidx; j < numInputs;j += gridDim.x * blockDim.x) {
		for (int z = tidy; z < numOutputs;z += gridDim.y * blockDim.y) {
			weights[(z * numInputs) + j] = randomNumbers[(z * numInputs) + j];
			if (tidx == 0) {
				biases[z] = 0;
			}
		}
	}
}

// cpu code for initializing in case gpu code doesnt work out
void initializeWeightsBiases(double* weights, double* biases, int numInputs, int numOutputs) {

}

// reading weights into gpu and returning the gpu allocated struct, we can read layer by layer to save on host memory
// if a particular layer is not defined, we will initialize it with random weights and 0 biases
GPUNet* loadGPUNet() {
	GPUNet* hostStruct = (GPUNet*)malloc(sizeof(GPUNet));
	int numLayers = standardNetSize;
	hostStruct->numLayers = numLayers;
	// allocating outer memory on cpu (inner memory will be allocated on gpu)
	double** hostWeights, **hostBiases, **hostWeightAdjustments, **hostLayerInput;
	int* numInputs, * numOutputs;
	hostWeights = (double**) malloc(sizeof(double*)*numLayers);
	hostBiases = (double**) malloc(sizeof(double*)*numLayers);
	hostWeightAdjustments = (double**) malloc(sizeof(double*)*numLayers);
	hostWeightAdjustments = (double**) malloc(sizeof(double*)*numLayers);
	hostLayerInput= (double**) malloc(sizeof(double*)*numLayers);
	numInputs = (int*)malloc(sizeof(int) * numLayers);
	numOutputs = (int*)malloc(sizeof(int) * numLayers);

	//setting allocating memory to struct
	hostStruct->weights = hostWeights;
	hostStruct->biases = hostBiases;
	hostStruct->weightAdjustments = hostWeightAdjustments;
	hostStruct->layerInput = hostLayerInput;
	hostStruct->numInputs = numInputs;
	hostStruct->numOutputs = numOutputs;
	
	string weightName = string("weights.txt");
	// dimension for 2D kernel launches
	dim3 blockShape(16, 16);
	dim3 gridShape(32, 32);
	for (int i = 0;i < numLayers;i++) {
		//adjusting the weight name
		weightName.insert(weightName.begin(), '0' + i);
		// setting our input and output sizes
		int specificInputSize;
		int specificOutputSize;
		if (i == 0) {
			specificInputSize = inputSize;
			specificOutputSize = hiddenLayerNumNeurons;
		}
		else if (i == numLayers - 1) {
			specificInputSize = hiddenLayerNumNeurons;
			specificOutputSize = outputSize;
		}
		else {
			// then this is hidden layer to hidden layer
			specificInputSize = hiddenLayerNumNeurons;
			specificOutputSize = hiddenLayerNumNeurons;
		}
		//setting layer size
		numInputs[i] = specificInputSize;
		numOutputs[i] = specificOutputSize;
		
		// allocating inner memory on gpu
		double* innerDeviceWeights, *deviceInnerBiases, *deviceInnerWeightAdjustments, *deviceInnerLayerInput;
		cudaErrorCheck(cudaMalloc(&innerDeviceWeights, sizeof(double) * specificInputSize * specificOutputSize));
		cudaErrorCheck(cudaMalloc(&deviceInnerWeightAdjustments, sizeof(double) * specificInputSize * specificOutputSize));
		cudaErrorCheck(cudaMalloc(&deviceInnerBiases, sizeof(double) * specificOutputSize));
		cudaErrorCheck(cudaMalloc(&deviceInnerLayerInput, sizeof(double) * specificInputSize));
		// seeing if we have a weights file for this
		// if not, then we can call the initialize kernel here
		FILE* weightsFile = fopen(weightName.c_str(), "r");
		if (weightsFile == NULL) {
			// then weights dont exist
			// getting random buffer for the allocation kernel
			curandGenerator_t gen;
			double* randomBuffer;
			cudaErrorCheck(cudaMalloc(&randomBuffer, sizeof(double) * specificInputSize * specificOutputSize));
			// seeding generator
			curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
			curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
			// getting random numbers
			curandGenerateUniformDouble(gen, randomBuffer, specificInputSize * specificOutputSize);
			// we will call the initialize kernel
			initializeGPUNet <<<gridShape, blockShape >>>(innerDeviceWeights, randomBuffer, deviceInnerBiases, specificInputSize, specificOutputSize);
			// freeing memory and destorying curand generator
			cudaErrorCheck(cudaFree(randomBuffer));
			curandDestroyGenerator(gen);
		}
		else {
			// then we will read in the weights and biases from a file and then copy them to the gpu
			double* hostWeights, * hostBiases;
			hostWeights = (double*)malloc(sizeof(double) * specificInputSize * specificOutputSize);
			hostBiases = (double*)malloc(sizeof(double) * specificOutputSize);
			//reading the weights
			for (int i = 0;i < specificOutputSize;i++) {
				for (int j = 0;j < specificInputSize;j++) {
					if (j == specificInputSize-1) {
						fscanf(weightsFile, "%lf\n", &(hostWeights[(i * specificInputSize) + j]));
					}
					else {
						fscanf(weightsFile, "%lf ", &(hostWeights[(i * specificInputSize) + j]));
					}
				}
			}
			//reading in the biases 
			for (int i = 0;i < specificOutputSize;i++) {
				fscanf(weightsFile, "%lf\n", &(hostBiases[i]));
			}
			//copying the weights and biases to gpu and freeing host memory
			cudaErrorCheck(cudaMemcpy(innerDeviceWeights, hostWeights, sizeof(double) * specificInputSize * specificOutputSize, cudaMemcpyHostToDevice));
			cudaErrorCheck(cudaMemcpy(deviceInnerBiases, hostBiases, sizeof(double) *  specificOutputSize, cudaMemcpyHostToDevice));
			free(hostWeights);
			free(hostBiases);

		}
		// closing the filestream and resetting the string
		if (weightsFile != NULL) fclose(weightsFile);
		weightName = string("weights.txt");

		// setting the inner pointers to the gpu allocated memory
		hostWeights[i] = innerDeviceWeights;
		hostBiases[i] = deviceInnerBiases;
		hostWeightAdjustments[i] = deviceInnerWeightAdjustments;
		hostLayerInput[i] = deviceInnerLayerInput;
	}

	// returning the struct containing gpu allocated pointer
	return hostStruct;
}


// writing weights from gpu to filesystem, we can write layer by layer to save on host memory
void writeGPUNet(GPUNet* net) {
	string weightFile = string("weights.txt");
	for (int i = 0;i < net->numLayers;i++) {
		//setting up file to be written to (overwritten)
		weightFile.insert(weightFile.begin(), '0' + i);
		remove(weightFile.c_str());
		FILE* toWrite = fopen(weightFile.c_str(), "w");
		// setting up host memory as intermediate for writing to file
		double* hostWeights, * hostBiases;
		hostWeights = (double*)malloc(sizeof(double) * net->numInputs[i] * net->numOutputs[i]);
		hostBiases = (double*)malloc(sizeof(double) * net->numOutputs[i]);
		// copying the weights and biases from gpu memory to host memory
		cudaErrorCheck(cudaMemcpy(hostWeights, net->weights[i], sizeof(double) * net->numInputs[i] * net->numOutputs[i], cudaMemcpyDeviceToHost));
		cudaErrorCheck(cudaMemcpy(hostBiases, net->biases[i], sizeof(double) *  net->numOutputs[i], cudaMemcpyDeviceToHost));

		// writing with fprint
		//writing weights first
		for (int j = 0;j < net->numOutputs[i];j++) {
			for (int z = 0;z < net->numInputs[i]; z++) {
				if (z == net->numInputs[i] - 1) {
					fprintf(toWrite, "%.15lf\n", hostWeights[(j*net->numInputs[i])+z]);
				}
				else {
					fprintf(toWrite, "%.15lf ", hostWeights[(j*net->numInputs[i])+z]);
				}
			}
		}

		//writing biases
		for (int j = 0;j < net->numOutputs[i];j++) {
			fprintf(toWrite, "%.15lf\n", hostBiases[j]);
		}

		// freeing host memory,closing file, and resetting weight file name
		free(hostWeights);
		free(hostBiases);
		fclose(toWrite);
		weightFile = string("weights.txt");
	}
}

// does the initial step of backpropogation while evaluating (sets the adjustments to be the derivative values) (we are using sigmoid, so the derivative is just output(1-output)
__global__ void backPropogateGPUInputHelper(double* weightAdjustment, double* outputs, int numInputs, int numOutputs) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	for (int i = tidx;i < numOutputs; i += blockDim.x * gridDim.x) {
		for (int j = tidy;j < numInputs;j += blockDim.y * gridDim.y) {
			weightAdjustment[(i * numInputs) + j] = outputs[i] * (1-outputs[i]);
		}
	}
}

// evaluating the entire gpu net with cublas and some input
void evaluateGPUNet(GPUNet* toEvaluate, double* inputs, double* outputBuffer) {
	cudaErrorCheck(cudaMemcpy(toEvaluate->layerInput[0], inputs, sizeof(double) * toEvaluate->numInputs[0], cudaMemcpyHostToDevice));
	// going through every layer and applying cublas
	//wrapping multiplication with cublas	
	/*
	int* numInputs = (int*) malloc(sizeof(int) * toEvaluate->numLayers);
	int* numOutputs = (int*)malloc(sizeof(int) * toEvaluate->numLayers);
	cudaErrorCheck(cudaMemcpy(numInputs, toEvaluate->numInputs, sizeof(int) * toEvaluate->numLayers, cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaMemcpy(numOutputs, toEvaluate->numOutputs, sizeof(int) * toEvaluate->numLayers, cudaMemcpyDeviceToHost));
	*/

	//layer input and output to keep track of 
	double* layerOutput;

	// sizes for 2d kernels
	dim3 blockShape(16, 16);
	dim3 gridShape(32, 32);
	
	//initializing cublas handle and setting matrices
	cublasHandle_t handle;
	cublasStatus_t status;
	cublasCreate_v2(&handle);
	// looping for gpu multiplication and addition of layers
	for (int i = 0;i < toEvaluate->numLayers;i++) {
		
		int m = 1;
		int k = toEvaluate->numInputs[i];
		int n = toEvaluate->numOutputs[i];
		double identityScalar = 1.0;

		

		//allocating buffer for the input
		/*
		cudaErrorCheck(cudaMalloc((&layerInput), sizeof(double) * toEvaluate->numInputs[i]));
		if (i == 0) {
			cudaErrorCheck(cudaMemcpy(layerInput, inputs, sizeof(double) * toEvaluate->numOutputs[0], cudaMemcpyHostToDevice));
		}
		else {
			cudaErrorCheck(cudaMemcpy(layerInput, layerOutput, sizeof(double) * toEvaluate->numInputs[i], cudaMemcpyDeviceToDevice));
			cudaErrorCheck(cudaFree(layerOutput));
		} */

		//setting up output as a copy of biases
		cudaErrorCheck(cudaMalloc(&layerOutput, sizeof(double) * toEvaluate->numOutputs[i]));
		cudaErrorCheck(cudaMemcpy(layerOutput, toEvaluate->biases[i], sizeof(double) * toEvaluate->numOutputs[i], cudaMemcpyDeviceToDevice));
		
		//calling cublas matrix multiply and adding biases vector (this does deviceWeights*deviceInputs + biasVector) and stores the result in the layerOutput vector

		status = cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &identityScalar, toEvaluate->layerInput[i], m, toEvaluate->weights[i], k, &identityScalar, layerOutput, m);

		if (status != CUBLAS_STATUS_SUCCESS) {
			printf("error with cublas matrix multiplication\n");
		}

		// applying sigmoid to the output
		
		sigmoidMatrix <<<20,512 >>> (layerOutput, toEvaluate->numOutputs[i]);
		cudaErrorCheck(cudaGetLastError());

		//double* sigmoidedCheck = (double*)malloc(sizeof(double) * toEvaluate->numInputs[i] * toEvaluate->numOutputs[i]);
		//cudaErrorCheck(cudaMemcpy(sigmoidedCheck,layerOutput,sizeof(double)*toEvaluate->numOutputs[i]))
		

		/*
		double* toSigmoid = (double*)malloc(sizeof(double) * toEvaluate->numInputs[i] * toEvaluate->numOutputs[i]);
		cudaMemcpy(toSigmoid, layerOutput, sizeof(double) * toEvaluate->numInputs[i] * toEvaluate->numOutputs[i], cudaMemcpyDeviceToHost);
		cpuSigmoidMatrix(toSigmoid, toEvaluate->numInputs[i] * toEvaluate->numOutputs[i]);
		cudaMemcpy(layerOutput, toSigmoid, sizeof(double) * toEvaluate->numInputs[i] * toEvaluate->numOutputs[i], cudaMemcpyHostToDevice);
		free(toSigmoid);
		*/

		//freeing device memory
		if (i == toEvaluate->numLayers - 1) {
			//copying output to final buffer
			cudaErrorCheck(cudaMemcpy(outputBuffer, layerOutput, sizeof(double) * toEvaluate->numOutputs[i], cudaMemcpyDeviceToHost));
		}
		else {
			//copying output to next layers input
			cudaErrorCheck(cudaMemcpy(toEvaluate->layerInput[i + 1], layerOutput, sizeof(double) * toEvaluate->numOutputs[i], cudaMemcpyDeviceToDevice));
		}
		//doing the propogation helper step
		backPropogateGPUInputHelper <<<gridShape, blockShape>>> (toEvaluate->weightAdjustments[i], layerOutput, toEvaluate->numInputs[i], toEvaluate->numOutputs[i]);
		cudaErrorCheck(cudaGetLastError());
		cudaErrorCheck(cudaFree(layerOutput));

	}
	//destroying handle
	cublasDestroy_v2(handle);
}



// calculates the adjustments based on the derivatives and sets up derivatives for next layer of backpropogation
__global__ void weightAdjust(double* weightAdjustments,double* biases, double* weights, double* derivatives, double* nextDerivatives, int numInputs, int numOutputs, double learningRate) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	for (int i = tidx; i < numOutputs;i += gridDim.x * blockDim.x) {
		for (int j = tidy;j < numInputs;j += gridDim.y * blockDim.y) {
			weightAdjustments[(i * numInputs) + j] *= derivatives[i];
			atomicAdd(&(nextDerivatives[j]),weightAdjustments[(i * numInputs) + j] * weights[(i*numInputs)+j] );
		}
		if (tidy == 0) {
			biases[i] -= learningRate * weightAdjustments[(i * numInputs)];
		}
	}
}

//actually adjusts the weights and biases
__global__ void finalizeWeightAdjust(double* weights, double* weightAdjustments, double* inputs, int numInputs, int numOutputs,double learningRate) {
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	for (int i = tidx;i < numOutputs;i += gridDim.x * blockDim.x) {
		for (int j = tidy;j < numInputs; j+= gridDim.y*blockDim.y) {
			weights[(i*numInputs) + j] -= learningRate * weightAdjustments[(i*numInputs)+j] * inputs[j];
		}
	}
}

// backpropogation of the entire net given the output values and the actual values
void backPropogateGPUNet(GPUNet* toBackProp, double* outputBuffer, double* actualRed, double* actualGreen, double* actualBlue, double learningRate) {
	// going through each layer and setting weight adjustments	
	// then performing the adjustments
	double* derivatives = (double*)malloc(sizeof(double) * outputSize);
	double* nextDerivatives;
	//setting the initial partial derivatives
	for (int i = 0;i < outputSize;i+=3) {
		derivatives[i]=(-(actualRed[i] - outputBuffer[i]));
		derivatives[i+1]=(-(actualGreen[i] - outputBuffer[i+1]));
		derivatives[i+2]=(-(actualBlue[i] - outputBuffer[i+2]));
	}

	dim3 dimBlock(16, 16);
	dim3 dimGrid;
	dimGrid.x = 32;
	dimGrid.y = 32;

	//copying derivatives to deviceDerivatives pointer
	double* deviceDerivatives;
	cudaMalloc(&deviceDerivatives, sizeof(double) * outputSize);
	cudaErrorCheck(cudaMemcpy(deviceDerivatives, derivatives, sizeof(double) * outputSize, cudaMemcpyHostToDevice));

	for (int z = toBackProp->numLayers-1;z >= 0;z--) {
		cudaErrorCheck(cudaMalloc(&nextDerivatives, sizeof(double) * toBackProp->numInputs[z]));
		weightAdjust<<<dimGrid,dimBlock>>>(toBackProp->weightAdjustments[z], toBackProp->biases[z], toBackProp->weights[z], deviceDerivatives,nextDerivatives, toBackProp->numInputs[z], toBackProp->numOutputs[z], learningRate);
		//freeing memory and setting up for next iteration
		cudaErrorCheck(cudaFree(deviceDerivatives));
		deviceDerivatives = nextDerivatives;
	}

	cudaErrorCheck(cudaFree(deviceDerivatives));

	//actually doing the weight adjustments
	for (int z = 0;z < toBackProp->numLayers;z++) {	
		finalizeWeightAdjust<<<dimGrid,dimBlock>>>(toBackProp->weights[z], toBackProp->weightAdjustments[z], toBackProp->layerInput[z],toBackProp->numInputs[z] ,toBackProp->numOutputs[z], learningRate);
	}

	// freeing host memory
	free(derivatives);
	
}

// given an image, we will run the net on it and output the result image
void outputFromGPUNet(char* imageName) {
	//will divide the image into squares and output it
}


// gpu trains the neural net on a random image from the dataset given a learning rate
void trainFromDataSet(double learningRate) {
	// loading the net
	GPUNet* toTrain = loadGPUNet();
	
	size_t freeMem;
	size_t totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	cout << "total memory of gpu: " << totalMem << "\n";
	cout << "total free memory of gpu after loading net: " << freeMem << "\n";
	

	
	while (true) {
		
		// pick a random image from the training dataset
		CImg<int> randomImage = getRandomTrainingImage();
		// converting image to black and white
		int* bwBuffer = (int*)malloc(sizeof(int) * randomImage.height() * randomImage.width());
		makeImageBlackAndWhiteWrapper(randomImage.data(), randomImage.data() + (randomImage.height() * randomImage.width()), randomImage.data() + (2 * randomImage.height() * randomImage.width()), bwBuffer, randomImage.height(), randomImage.width());
		// crop parts to fit neural net input size
		// we will crop into perfect squares and then combine them to get the final image (we do not need to combine them for training though)
		int trainCount = 0;
		//R,G,B training errors
		double currTrainingError[3]{};
		while (trainCount != epochNum) {
			for (int i = 0;i < randomImage.height();i += squareSide) {
				for (int j = 0;j < randomImage.width();j += squareSide) {
					// getting the square for both the bw image and color image
					int* bwSquare = (int*)malloc(sizeof(int) * squareSide * squareSide);
					int* redSquare = (int*)malloc(sizeof(int) * squareSide * squareSide);
					int* greenSquare = (int*)malloc(sizeof(int) * squareSide * squareSide);
					int* blueSquare = (int*)malloc(sizeof(int) * squareSide * squareSide);
					getSquareWrapper(bwBuffer, bwSquare, squareSide, randomImage.height(), randomImage.width(), i, j);
					getSquareWrapper(randomImage.data(), redSquare, squareSide, randomImage.height(), randomImage.width(), i, j);
					getSquareWrapper(randomImage.data() + (randomImage.height() * randomImage.width()), greenSquare, squareSide, randomImage.height(), randomImage.width(), i, j);
					getSquareWrapper(randomImage.data() + (2 * randomImage.height() * randomImage.width()), blueSquare, squareSide, randomImage.height(), randomImage.width(), i, j);

					// scale pixels by 255 for both bw image and color image
					double* scaledBWSquare = (double*)malloc(sizeof(double) * squareSide * squareSide);
					pixelScaleWrapper(bwSquare, scaledBWSquare, squareSide, squareSide);
					free(bwSquare);
					double* scaledRedSquare = (double*)malloc(sizeof(double) * squareSide * squareSide);
					pixelScaleWrapper(redSquare, scaledRedSquare, squareSide, squareSide);
					free(redSquare);
					double* scaledGreenSquare = (double*)malloc(sizeof(double) * squareSide * squareSide);
					pixelScaleWrapper(greenSquare, scaledGreenSquare, squareSide, squareSide);
					free(greenSquare);
					double* scaledBlueSquare = (double*)malloc(sizeof(double) * squareSide * squareSide);
					pixelScaleWrapper(blueSquare, scaledBlueSquare, squareSide, squareSide);
					free(blueSquare);

					// evaluate net for each part of the image
					// we will have outputSize number of outputs, and we will train the net so that the output[0] is first pixels R, output[1],output[2] represent first pixels G and B value
					double* outputBuffer = (double*)malloc(sizeof(double) * outputSize);
					
					/*
					cudaMemGetInfo(&freeMem, &totalMem);
					cout << "total free memory of gpu before evaluating net: " << freeMem << "\n";
					*/

					evaluateGPUNet(toTrain, scaledBWSquare, outputBuffer);

					/*
					cudaMemGetInfo(&freeMem, &totalMem);
					cout << "total free memory of gpu after evaluating net: " << freeMem << "\n";
					*/

					// printing out the current training error
					for (int i = 0;i < outputSize;i += 3) {
						currTrainingError[0] += pow(scaledRedSquare[i] - outputBuffer[i], 2);
						currTrainingError[1] += pow(scaledGreenSquare[i] - outputBuffer[i + 1], 2);
						currTrainingError[2] += pow(scaledBlueSquare[i] - outputBuffer[i + 2], 2);
					}
					/*
					cout << "red error: " << currTrainingError[0] << "\n";
					cout << "green error: " << currTrainingError[1] << "\n";
					cout << "blue error: " << currTrainingError[2] << "\n";
					*/

					//incrementTrainingErrorGPU()
					/*
					cudaMemGetInfo(&freeMem, &totalMem);
					cout << "total free memory of gpu before backpropogation: " << freeMem << "\n";
					*/

					backPropogateGPUNet(toTrain, outputBuffer, scaledRedSquare, scaledGreenSquare, scaledBlueSquare, learningRate);
					
					/*
					cudaMemGetInfo(&freeMem, &totalMem);
					cout << "total free memory of gpu after backpropogation: " << freeMem << "\n";
					*/

					// freeing memory we no longer need
					free(scaledBWSquare);
					free(scaledRedSquare);
					free(scaledGreenSquare);
					free(scaledBlueSquare);
					free(outputBuffer);
				}
			}
			trainCount++;
			cout << "FINISHED EPOCH: " << trainCount << "\n";
		}

		//freeing allocated memory
		free(bwBuffer);
		// then we will perform a test of error on a random test data image
		// writing weights back to filesystem now that the epochLimit was reached
		cout << "writing updated weights to filesystem\n";
		writeGPUNet(toTrain);
	}	
}

// gpu tests the neural net error on a specific image
void testImage(char* imageName) {
	//grabbing image

	//cropping parts to fit neural net input size

	//scale pixels by 255

	//evaluate net for each part

	//calculate error
}

