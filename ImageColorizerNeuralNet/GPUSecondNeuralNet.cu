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



using namespace cimg_library;
using namespace std;



// my idea here is to create a more memory hitting neural net in order to flex the gpu more and speedup training/evaluating and output times
// idea is to take a 1000 x 1000 portion of the image, and guess the middle 500x500 pixel values so the output will be of size 500 x 500 x 3 for RGB values
	// this way, we can extend smaller images with black values to apply to the model and we can batch portions of larger images more easily
	// I will test training and if I cannot get a fit (which is likely with only 100 neurons a layer, I will try increasing the number of neurons)
	// if there is still sufficient gpu memory not being utilized, I can increase the input size and the output size and the number of neurons per layer to get a good mix

CImg<int> getRandomTestImage() {
	string searchName = "./TestData/*";
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;
	int numPictures = 0;
	hFind = FindFirstFile(searchName.c_str(), &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		cout << "OH NO Test DATA NOT FOUND!\n";
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
		cout << "OH NO!!!!! We have an empty test set! testing aborted\n";
		//returning empty image
		FindClose(hFind);
		return CImg<int>();
	}
	FindClose(hFind);
	// idea is to choose a random picture in the training data folder
	// getting random number in a range
	random_device rando;
	mt19937 gen(rando());
	uniform_int_distribution<> distr(1, numPictures);
	int imageToPick = distr(gen);

	// choosing a random picture
	WIN32_FIND_DATA RandomFileData;
	HANDLE hFindRandom = NULL;

	int currCount = 0;
	string dirName = "TestData/";
	while (currCount != imageToPick) {
		if (hFindRandom == NULL) {
			hFindRandom = FindFirstFile("./TestData/*\0", &RandomFileData);
		}
		else {
			FindNextFile(hFindRandom, &RandomFileData);
		}
		if ((RandomFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
			currCount++;
		}
	}
	// now the data is pointed to the picture to choose

	// using cimg to get data for the given picture
	for (int i = 0;i < strlen(RandomFileData.cFileName);i++) {
		dirName.push_back(RandomFileData.cFileName[i]);
	}
	CImg<int> colorPicture(dirName.c_str());
	cout << "grabbed test image: " << dirName << "\n";
	FindClose(hFindRandom);
	return colorPicture;
}

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
	FindClose(hFind);
	// idea is to choose a random picture in the training data folder
	// getting random number in a range
	random_device rando;
	mt19937 gen(rando());
	uniform_int_distribution<> distr(1, numPictures);
	int imageToPick = distr(gen);

	// choosing a random picture
	WIN32_FIND_DATA RandomFileData;
	HANDLE hFindRandom = NULL;

	int currCount = 0;
	string dirName = "TrainingData/";
	while (currCount != imageToPick) {
		if (hFindRandom == NULL) {
			hFindRandom = FindFirstFile("./TrainingData/*\0", &RandomFileData);
		}
		else {
			FindNextFile(hFindRandom, &RandomFileData);
		}
		if ((RandomFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
			currCount++;
		}
	}
	// now the data is pointed to the picture to choose

	// using cimg to get data for the given picture
	for (int i = 0;i < strlen(RandomFileData.cFileName);i++) {
		dirName.push_back(RandomFileData.cFileName[i]);
	}
	CImg<int> colorPicture(dirName.c_str());
	cout << "grabbed training image: " << dirName << "\n";
	FindClose(hFindRandom);
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

__global__ void reluMatrix(double* output, int dim) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i = tidx;i < dim;i += gridDim.x * blockDim.x) {
		output[i] = fmax(0.0, output[i]);
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
		output[i] = 1 / (1 + exp(-(output[i])));
	}
}

// given weight matrices and biases , we can initialize weights with random small numbers and biases with very small static values
__global__ void initializeGPUNet(double* weights, double* randomNumbers, double* biases, int numInputs, int numOutputs) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	for (int j = tidx; j < numInputs;j += gridDim.x * blockDim.x) {
		for (int z = tidy; z < numOutputs;z += gridDim.y * blockDim.y) {
			weights[(z * numInputs) + j] = randomNumbers[(z * numInputs) + j]/(numInputs);
			//weights[(z * numInputs) + j] = 0;
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
	double** hostWeights, **hostBiases, **hostWeightAdjustments, **hostLayerInput, ** hostDeltas;
	int* numInputs, * numOutputs;
	hostWeights = (double**) malloc(sizeof(double*)*numLayers);
	hostBiases = (double**) malloc(sizeof(double*)*numLayers);
	//hostWeightAdjustments = (double**) malloc(sizeof(double*)*numLayers);
	hostDeltas = (double**)malloc(sizeof(double*) * numLayers);
	hostLayerInput= (double**) malloc(sizeof(double*)*numLayers);
	numInputs = (int*)malloc(sizeof(int) * numLayers);
	numOutputs = (int*)malloc(sizeof(int) * numLayers);

	//setting allocating memory to struct
	hostStruct->weights = hostWeights;
	hostStruct->biases = hostBiases;
	//hostStruct->weightAdjustments = hostWeightAdjustments;
	hostStruct->deltas = hostDeltas;
	hostStruct->layerInput = hostLayerInput;
	hostStruct->numInputs = numInputs;
	hostStruct->numOutputs = numOutputs;
	
	string weightName = string("weights.txt");
	// dimension for 2D kernel launches
	dim3 blockShape(32, 32);
	dim3 gridShape(16, 16);
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
		double* innerDeviceWeights, *deviceInnerBiases, *deviceInnerWeightAdjustments, *deviceInnerLayerInput, *deviceInnerDeltas;
		cudaErrorCheck(cudaMalloc(&innerDeviceWeights, sizeof(double) * specificInputSize * specificOutputSize));
		//cudaErrorCheck(cudaMalloc(&deviceInnerWeightAdjustments, sizeof(double) * specificInputSize * specificOutputSize));
		cudaErrorCheck(cudaMalloc(&deviceInnerDeltas, sizeof(double)  * specificOutputSize));
		cudaErrorCheck(cudaMalloc(&deviceInnerBiases, sizeof(double) * specificOutputSize));
		cudaErrorCheck(cudaMalloc(&deviceInnerLayerInput, sizeof(double) * specificInputSize * numInputSquares));
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
			curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
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
			for (int k = 0;k < specificOutputSize;k++) {
				for (int j = 0;j < specificInputSize;j++) {
					if (j == specificInputSize-1) {
						int count = 0;
						do {
							count =fscanf(weightsFile, "%lf\n", &(hostWeights[(k * specificInputSize) + j]));
						} while (count == 0);
					}
					else {
						int count = 0;
						do {
							count = fscanf(weightsFile, "%lf ", &(hostWeights[(k * specificInputSize) + j]));
						} while (count == 0);
					}
				}
			}
			//reading in the biases 
			for (int k = 0;k < specificOutputSize;k++) {
				int count = 0;
				do {
					count =fscanf(weightsFile, "%lf\n", &(hostBiases[k]));
				} while (count == 0);
			}
			//copying the weights and biases to gpu and freeing host memory
			cudaErrorCheck(cudaMemcpy(innerDeviceWeights, hostWeights, sizeof(double) * specificInputSize * specificOutputSize, cudaMemcpyHostToDevice));
			cudaErrorCheck(cudaMemcpy(deviceInnerBiases, hostBiases, sizeof(double) *  specificOutputSize, cudaMemcpyHostToDevice));
			free(hostWeights);
			free(hostBiases);
			cout << "loaded existing weights for layer: " << i << " \n";

		}
		// closing the filestream and resetting the string
		if (weightsFile != NULL) fclose(weightsFile);
		weightName = string("weights.txt");

		// setting the inner pointers to the gpu allocated memory
		hostWeights[i] = innerDeviceWeights;
		hostBiases[i] = deviceInnerBiases;
		//hostWeightAdjustments[i] = deviceInnerWeightAdjustments;
		hostLayerInput[i] = deviceInnerLayerInput;
		hostDeltas[i] = deviceInnerDeltas;
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
		//ofstream weightText;
		//weightText.open(weightFile, ios::trunc | ios::out);
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
					int count = 0;
					do {
						count = fprintf(toWrite, "%.15lf\n", hostWeights[(j * net->numInputs[i]) + z]);
					} while (count == 0);
				}
				else {
					int count = 0;
					do {
						count = fprintf(toWrite, "%.15lf ", hostWeights[(j * net->numInputs[i]) + z]);
					} while (count == 0);
				}
			}
		}

		//writing biases
		for (int j = 0;j < net->numOutputs[i];j++) {
			int count = 0;
			do { 
				count = fprintf(toWrite, "%.15lf\n", hostBiases[j]);
			} while (count == 0);
		}

		// freeing host memory,closing file, and resetting weight file name
		free(hostWeights);
		free(hostBiases);
		fclose(toWrite);
		weightFile = string("weights.txt");
	}
}

// keep in mind we might have multiple inputs, so we can take the average gradient towards the end
// this kernel is agnostic to the number of input patches to the neural net
__global__ void backPropogationGradientCalculationOutputLayer(double* actualOutputs, double* netOutputs, double* outputDeltas, int numOutputs) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = tid;i < numOutputs; i += gridDim.x * blockDim.x) {
		//sigmoid derivative
		for (int j = 0;j < numInputSquares;j++) {
			
			outputDeltas[i] += (netOutputs[(i*numInputSquares)+j]-actualOutputs[(i*numInputSquares)+j]) * netOutputs[(i*numInputSquares)+j] * (1 - netOutputs[(i*numInputSquares)+j]);
		}
		//relu derivative
		/*
		if (netOutputs[i] == 0) {
			
			outputDeltas[i] = (netOutputs[i]-actualOutputs[i]) * 0;
		}
		else {
			outputDeltas[i] = (netOutputs[i]-actualOutputs[i]) * 1;
		}
		*/
	}
}

__global__ void backPropogationGradientCalculationHiddenLayer(double* layerOutput, double* outputDeltas, double* nextLayerWeights, double* nextLayerDeltas, int numLayerOutputs, int numNextLayerOutputs) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//firstly getting sum delta of associated successive layer, and then multiplying derivative as well
	for (int i = tid;i < numLayerOutputs; i += gridDim.x * blockDim.x) {
		for (int j = 0;j < numNextLayerOutputs;j++) {
			//delta calculation
			//sigmoid derivative
			outputDeltas[i] += nextLayerDeltas[j] * nextLayerWeights[(j * numLayerOutputs) + i] * layerOutput[i] * (1-layerOutput[i]);
			//relu derivative
			/*
			if (layerOutput[i] == 0) {
					outputDeltas[i] += nextLayerDeltas[j] * nextLayerWeights[(j * numLayerOutputs) + i] * 0;
			}
			else {
					outputDeltas[i] += nextLayerDeltas[j] * nextLayerWeights[(j * numLayerOutputs) + i] * 1;
			} */
			
			
		}
	}
}

//function for getting average deltas for the updates
// num outputs is the output size of the net itself on a single input
// num inputs is the number of input patches given to the net
__global__ void averageBatchGradient(double* gradient, int numInputs, int numOutputs) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = tid;i < numOutputs;i+=gridDim.x*blockDim.x) {
		// average calculation
		gradient[i] /= numInputs;
	}
}

//function that actually updates the weights based on the deltas, we will just use the last input as the basis for backtracking
__global__ void finalizeUpdate(double* layerInputs, double* layerDeltas, double* layerBiases, double learningRate, double* weights, int numOutputs, int numInputs) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = tid;i < numOutputs;i+=gridDim.x*blockDim.x) {
		for (int j = 0;j < numInputs;j++) {
		
			// just arbitrarily picking the first input to use for backprop updating
			weights[(i * numInputs) + j] -= learningRate * layerDeltas[i] * layerInputs[(j*numInputSquares)];
		}
		layerBiases[i] -= learningRate * layerDeltas[i];
	}
}

void batchBackPropogation(GPUNet* toTrain,double* finalOutput,double* netOutput,double learningRate) {
	double* deviceFinalOutput, *deviceNetOutput;
	cudaErrorCheck(cudaMalloc(&deviceFinalOutput, sizeof(double) * outputSize * numInputSquares));
	cudaErrorCheck(cudaMalloc(&deviceNetOutput, sizeof(double) * outputSize * numInputSquares));
	cudaErrorCheck(cudaMemcpy(deviceNetOutput, netOutput, sizeof(double) * outputSize * numInputSquares, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(deviceFinalOutput, finalOutput, sizeof(double) * outputSize *numInputSquares , cudaMemcpyHostToDevice));
	for (int i = toTrain->numLayers - 1;i >= 0;i--) {
		//memsetting the delta for this layer to all zeroes
		cudaErrorCheck(cudaMemset(toTrain->deltas[i], 0, sizeof(double) * toTrain->numOutputs[i] ));
		if (i == toTrain->numLayers - 1) {
			// special kernel for output layer
			backPropogationGradientCalculationOutputLayer << <20, 256>> > (deviceFinalOutput,deviceNetOutput,toTrain->deltas[i],toTrain->numOutputs[i]);
			averageBatchGradient << <20, 256 >> > (toTrain->deltas[i], numInputSquares, toTrain->numOutputs[i]);
			// freeing device memory we no longer need
			cudaErrorCheck(cudaFree(deviceFinalOutput));
			cudaErrorCheck(cudaFree(deviceNetOutput));
		}
		else {
			// then we have a hidden layer
			backPropogationGradientCalculationHiddenLayer <<<20,256>>> (toTrain->layerInput[i+1],toTrain->deltas[i],toTrain->weights[i+1],toTrain->deltas[i+1],toTrain->numOutputs[i],toTrain->numOutputs[i+1]);
		}
	}

	// doing weight updates
	for (int i = 0;i < toTrain->numLayers;i++) {
		finalizeUpdate << <20, 256>> > (toTrain->layerInput[i], toTrain->deltas[i], toTrain->biases[i],learningRate, toTrain->weights[i], toTrain->numOutputs[i], toTrain->numInputs[i]);
	}
	// now weights are updated
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

//verifying that my cublas logic and cpu matrix logic are the same -> returns false if not the same
boolean testCublas() {
	// test weights is a 5x6 matrix with values from 1 to 30
	// test biases is a vector of output size with values 1 to 5
	// test inputs is a vector of inputs with values from  1 to 6
	int numInputs = 6;
	int numOutputs = 5;
	double* testWeights, * testBias, * testInput;
	testWeights = (double*)malloc(sizeof(double) * numInputs * numOutputs);
	testBias = (double*)malloc(sizeof(double) * numOutputs);
	testInput = (double*)malloc(sizeof(double) * numInputs);
	for (int i = 0;i < numInputs * numOutputs;i++) {
		if (i < 5) {
			testBias[i] = i + 1;
		}

		if (i < 6) {
			testInput[i] = i + 1;
		}

		testWeights[i] = i + 1;
	}
	// cuda logic
	double* deviceCPUTestOutput = (double*)malloc(sizeof(double) * numOutputs);
	double* deviceWeights, * deviceBiases, * deviceOutput, * deviceInput;
	cudaErrorCheck(cudaMalloc(&deviceWeights, sizeof(double) * numInputs * numOutputs));
	cudaErrorCheck(cudaMalloc(&deviceBiases, sizeof(double) *  numOutputs));
	cudaErrorCheck(cudaMalloc(&deviceOutput, sizeof(double) *  numOutputs));
	cudaErrorCheck(cudaMalloc(&deviceInput, sizeof(double) *  numInputs));
	
	cudaErrorCheck(cudaMemcpy(deviceWeights, testWeights, sizeof(double) * numInputs * numOutputs, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(deviceBiases, testBias, sizeof(double) *  numOutputs, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(deviceInput, testInput, sizeof(double) *  numInputs, cudaMemcpyHostToDevice));

	// copying biases to output buffer
	cudaErrorCheck(cudaMemcpy(deviceOutput, deviceBiases, sizeof(double) *  numOutputs, cudaMemcpyDeviceToDevice));

	cublasHandle_t handle;
	cublasStatus_t status;
	cublasCreate_v2(&handle);
		
	int m = 1;
	int k = numInputs;
	int n = numOutputs;
	double identityScalar = 1.0;

	//calling cublas matrix multiply and adding biases vector (this does deviceWeights*deviceInputs + biasVector) and stores the result in the layerOutput vector

	status = cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &identityScalar, deviceInput, m, deviceWeights, k, &identityScalar, deviceOutput, m);


	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("error with cublas matrix multiplication\n");
	}

	cudaErrorCheck(cudaMemcpy(deviceCPUTestOutput, deviceOutput, sizeof(double) * numOutputs, cudaMemcpyDeviceToHost));

	cudaErrorCheck(cudaFree(deviceWeights));
	cudaErrorCheck(cudaFree(deviceBiases));
	cudaErrorCheck(cudaFree(deviceOutput));
	cudaErrorCheck(cudaFree(deviceInput));	

	//destroying handle
	cublasDestroy_v2(handle);

	// doing cpu matrix multiplication and addition
	double* result = (double*)malloc(sizeof(double) * numOutputs);

	for (int k = 0;k < numOutputs;k++) {
		double sum = 0;
		for (int i = 0;i < numInputs;i++) {
			sum += testWeights[(k * numInputs) + i] * testInput[i];
		}
		result[k] = sum + testBias[k];
	}


	//verification
	for (int i = 0;i < numOutputs;i++) {
		if (deviceCPUTestOutput[i] != result[i]) {
			cout << "VERY BIG ISSUE!\n";
			return false;
		}
	}

	free(result);
	free(testWeights);
	free(testInput);
	free(testBias);

	return true;

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
	dim3 blockShape(32, 32);
	dim3 gridShape(16, 16);
	
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

		// applying sigmoid to the output layer and relu to the other layers
	
		if (i == toEvaluate->numLayers - 1) {
			sigmoidMatrix <<<20,512 >>> (layerOutput, toEvaluate->numOutputs[i]);
		}
		else {
			
			reluMatrix<<<20,512 >>> (layerOutput, toEvaluate->numOutputs[i]);
		}
		
		cudaErrorCheck(cudaGetLastError());

		//double* sigmoidedCheck = (double*)malloc(sizeof(double) * toEvaluate->numInputs[i] * toEvaluate->numOutputs[i]);
		//cudaErrorCheck(cudaMemcpy(sigmoidedCheck,layerOutput,sizeof(double)*toEvaluate->numOutputs[i]))
		
		/*
		double* toSigmoid = (double*)malloc(sizeof(double) * toEvaluate->numOutputs[i] );
		cudaMemcpy(toSigmoid, layerOutput, sizeof(double) * toEvaluate->numOutputs[i], cudaMemcpyDeviceToHost);
		cpuSigmoidMatrix(toSigmoid, toEvaluate->numOutputs[i]);
		cudaMemcpy(layerOutput, toSigmoid, sizeof(double) * toEvaluate->numOutputs[i], cudaMemcpyHostToDevice);
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
		//backPropogateGPUInputHelper <<<gridShape, blockShape>>> (toEvaluate->weightAdjustments[i], layerOutput, toEvaluate->numInputs[i], toEvaluate->numOutputs[i]);
		cudaErrorCheck(cudaGetLastError());
		cudaErrorCheck(cudaFree(layerOutput));

	}
	//destroying handle
	cublasDestroy_v2(handle);
}

// we can set the input to the be batched (i.e we keep the weights matrix, but the inputs will be a matrix instead of a vector, so we can calculate multiple outputs at the same time) 
// so, if we have a 1920x1080 image, our inputs will be a matrix where every column contains the 50x50 patch, and the output matrix will be a (3 x (1920x1080)) matrix for every pixels RGB value
// if this is too much, we can set a max batch input size, and then do a few different matrix multiplications -> max batch size defined in GPUSecondNeuralNet.cuh
void batchedGPUEvaluate(GPUNet* toEvaluate, double* inputs, double* outputBuffer) {
	cudaErrorCheck(cudaMemcpy(toEvaluate->layerInput[0], inputs, sizeof(double) * toEvaluate->numInputs[0]*numInputSquares, cudaMemcpyHostToDevice));
	// going through every layer and applying cublas
	//wrapping multiplication with cublas	
	//layer input and output to keep track of 
	double* layerOutput;

	// sizes for 2d kernels
	dim3 blockShape(32, 32);
	dim3 gridShape(16, 16);

	//initializing cublas handle and setting matrices
	cublasHandle_t handle;
	cublasStatus_t status;
	cublasCreate_v2(&handle);
	// looping for gpu multiplication and addition of layers
	for (int i = 0;i < toEvaluate->numLayers;i++) {

		int m = numInputSquares;
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

		//setting up output as a copy of biases in every column for our batched output
		cudaErrorCheck(cudaMalloc(&layerOutput, sizeof(double) * toEvaluate->numOutputs[i] * numInputSquares));
		double* biases = (double*)malloc(sizeof(double) * toEvaluate->numOutputs[i]);
		cudaErrorCheck(cudaMemcpy(biases, toEvaluate->biases[i], sizeof(double) * toEvaluate->numOutputs[i], cudaMemcpyDeviceToHost));
		double* biasMatrix = (double*)malloc(sizeof(double) * toEvaluate->numOutputs[i] * numInputSquares);
		for (int z = 0;z < numInputSquares; z++) {
			for (int y = 0;y < toEvaluate->numOutputs[i];y++) {
				biasMatrix[(y * numInputSquares) + z] = biases[y];
			}
		}
		//copying matrix to the cudaMallocedBuffer

		cudaErrorCheck(cudaMemcpy(layerOutput, biasMatrix, sizeof(double) * toEvaluate->numOutputs[i] * numInputSquares, cudaMemcpyHostToDevice));
		//freeing host memory
		free(biases);
		free(biasMatrix);
		

		//calling cublas matrix multiply and adding biases matrix (this does deviceWeights*deviceInputs + biasVector) and stores the result in the layerOutput matrix

		status = cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &identityScalar, toEvaluate->layerInput[i], m, toEvaluate->weights[i], k, &identityScalar, layerOutput, m);

		if (status != CUBLAS_STATUS_SUCCESS) {
			printf("error with cublas matrix multiplication\n");
		}

		// applying sigmoid to the output

		//sigmoidMatrix << <20, 512 >> > (layerOutput, toEvaluate->numOutputs[i] * numInputSquares);
		// applying sigmoid to the output layer and relu to the other layers
	
		//if (i == toEvaluate->numLayers - 1) {
			sigmoidMatrix <<<20,512 >>> (layerOutput, toEvaluate->numOutputs[i] * numInputSquares);
		//}
		//else {
			
		//	reluMatrix<<<20,512 >>> (layerOutput, toEvaluate->numOutputs[i]*numInputSquares);
		//}
		cudaErrorCheck(cudaGetLastError());

		//double* sigmoidedCheck = (double*)malloc(sizeof(double) * toEvaluate->numInputs[i] * toEvaluate->numOutputs[i]);
		//cudaErrorCheck(cudaMemcpy(sigmoidedCheck,layerOutput,sizeof(double)*toEvaluate->numOutputs[i]))

		/*
		double* toSigmoid = (double*)malloc(sizeof(double) * toEvaluate->numOutputs[i] );
		cudaMemcpy(toSigmoid, layerOutput, sizeof(double) * toEvaluate->numOutputs[i], cudaMemcpyDeviceToHost);
		cpuSigmoidMatrix(toSigmoid, toEvaluate->numOutputs[i]);
		cudaMemcpy(layerOutput, toSigmoid, sizeof(double) * toEvaluate->numOutputs[i], cudaMemcpyHostToDevice);
		free(toSigmoid);
		*/

		//freeing device memory
		if (i == toEvaluate->numLayers - 1) {
			//copying output to final buffer
			cudaErrorCheck(cudaMemcpy(outputBuffer, layerOutput, sizeof(double) * toEvaluate->numOutputs[i] * numInputSquares, cudaMemcpyDeviceToHost));
		}
		else {
			//copying output to next layers input
			cudaErrorCheck(cudaMemcpy(toEvaluate->layerInput[i + 1], layerOutput, sizeof(double) * toEvaluate->numOutputs[i] * numInputSquares, cudaMemcpyDeviceToDevice));
		}
		//doing the propogation helper step
		//backPropogateGPUInputHelper << <gridShape, blockShape >> > (toEvaluate->weightAdjustments[i], layerOutput, toEvaluate->numInputs[i], toEvaluate->numOutputs[i]);
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
		
	}
	__syncthreads();
	for (int i = tidx; i < numOutputs;i += gridDim.x * blockDim.x) {
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
			/*
			if (inputs[j] == 0) {
				printf("NOOOOOOOOOOOOOOO\n");
			} */
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
	for (int i = 0;i < outputSize;i++) {
		if (i < outputSquareSide * outputSquareSide) {
			//red
			derivatives[i]=(-(actualRed[i] - outputBuffer[i]));
		}
		else if (i < 2 * outputSquareSide * outputSquareSide) {
			//green
			derivatives[i]=(-(actualGreen[i-(outputSquareSide*outputSquareSide)] - outputBuffer[i]));
		}
		else {
			//blue
			derivatives[i]=(-(actualBlue[i-(2*outputSquareSide*outputSquareSide)] - outputBuffer[i]));
		}
		/*
		if (i % 3 == 0) {
			//red
			derivatives[i]=(-(actualRed[i/3] - outputBuffer[i]));
		}
		else if (i % 3 == 1) {
			//green
			derivatives[i]=(-(actualGreen[i/3] - outputBuffer[i]));
		}
		else {
			//blue
			derivatives[i]=(-(actualBlue[i/3] - outputBuffer[i]));
		}
		/*
		derivatives[i]=(-(actualRed[i] - outputBuffer[i]));
		derivatives[i+1]=(-(actualGreen[i] - outputBuffer[i+1]));
		derivatives[i+2]=(-(actualBlue[i] - outputBuffer[i+2]));
		*/
	}

	dim3 dimBlock(32, 32);
	dim3 dimGrid;
	dimGrid.x = 16;
	dimGrid.y = 16;

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

// given an image, we will run the net on it and output the result image, will also fill an error buffer
void outputFromGPUNet(char* imageName, char* outputImageName) {
	GPUNet* toTrain= loadGPUNet();
	double* layer2Weights = (double*)malloc(sizeof(double) * toTrain->numInputs[1] * toTrain->numOutputs[1]);
	cudaMemcpy(layer2Weights, toTrain->weights[1], sizeof(double) * toTrain->numInputs[1] * toTrain->numOutputs[1], cudaMemcpyDeviceToHost);
	cout << "layer 2 weight 40: " << layer2Weights[40] << " \n";
	free(layer2Weights);
	//will divide the image into squares and output it
	// going through patches, running the net on each patch, and then adding to the final buffer
	int imageCount = 0;
	//getting black and white image
	CImg<int> chosenImage(imageName);
	// converting image to black and white
	int* bwBuffer = chosenImage.data();
	int* finalBuffer = (int*)malloc(sizeof(int) * 3 * chosenImage.height() * chosenImage.width());
	//makeImageBlackAndWhiteWrapper(randomImage.data(), randomImage.data() + (randomImage.height() * randomImage.width()), randomImage.data() + (2 * randomImage.height() * randomImage.width()), bwBuffer, randomImage.height(), randomImage.width());
	// crop parts to fit neural net input size
	// we will crop into perfect squares and then combine them to get the final image (we do not need to combine them for training though)
		for (int i = 0;i < chosenImage.height();i += outputSquareSide) {
			for (int j = 0;j < chosenImage.width();j += outputSquareSide) {
				// getting the square for both the bw image and color image
				int* bwSquare = (int*)malloc(sizeof(int) * squareSide * squareSide);

				getSquareWrapper(bwBuffer, bwSquare, squareSide, chosenImage.height(), chosenImage.width(), i, j);

				
				CImg<int> testColorSquare(bwSquare, outputSquareSide, outputSquareSide);
				testColorSquare.save("outputSquare.jpg", imageCount);


				// scale pixels by 255 for both bw image and color image
				double* scaledBWSquare = (double*)malloc(sizeof(double) * squareSide * squareSide);
				pixelScaleWrapper(bwSquare, scaledBWSquare, squareSide, squareSide, inputPixelScaler);
				free(bwSquare);

				// evaluate net for each part of the image
				// we will have outputSize number of outputs, and we will train the net so that the output[0] is first pixels R, output[1],output[2] represent first pixels G and B value
				double* outputBuffer = (double*)malloc(sizeof(double) * outputSize);

				evaluateGPUNet(toTrain, scaledBWSquare, outputBuffer);

				
				double* copyOfOutput = (double*)malloc(sizeof(double) * outputSize);
				memcpy(copyOfOutput, outputBuffer, sizeof(double) * outputSize);
				for (int k = 0;k < outputSize;k++) {
					copyOfOutput[k] *= inputPixelScaler;
				}
				CImg<double> testBuffer(copyOfOutput, outputSquareSide, outputSquareSide, 1, 3);
				testBuffer.save("OutputSquareImage.jpg", imageCount);
				free(copyOfOutput);
				imageCount++;

			//copying output buffer to finalBuffer positions	
			for (int k = 0;k < outputSquareSide;k++) {
				for (int y = 0;y < outputSquareSide;y++) {
					if (i + k < chosenImage.height() && y + j < chosenImage.width()){
						finalBuffer[((i + k) * chosenImage.width()) + (j + y)] = outputBuffer[(k * outputSquareSide) + y]*inputPixelScaler;
						finalBuffer[(chosenImage.width() * chosenImage.height())+((i + k) * chosenImage.width()) + (j + y)] = outputBuffer[(outputSquareSide*outputSquareSide)+(k * outputSquareSide) + y]*inputPixelScaler;
						finalBuffer[(2*chosenImage.width() * chosenImage.height())+((i + k) * chosenImage.width()) + (j + y)] = outputBuffer[(2*outputSquareSide*outputSquareSide)+(k * outputSquareSide) + y]*inputPixelScaler;
					}
				}
			}	

			// freeing memory we no longer need
			free(scaledBWSquare);
			free(outputBuffer);
			}
		}
		// creating the color image and saving it to disk
		
		CImg<int> newColorImage(finalBuffer,  chosenImage.width(), chosenImage.height(), 1, 3);
		newColorImage.save(outputImageName);
		
}

// finish this method and randomize training
double testFromTestData(GPUNet* toTest) {
	// grabbing a random image
	CImg<int> randomImage = getRandomTestImage();
	// loading the net
	if (toTest== NULL) {
		toTest = loadGPUNet();
	}
	
	
	size_t freeMem;
	size_t totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	cout << "total memory of gpu: " << totalMem << "\n";
	cout << "total free memory of gpu after loading net: " << freeMem << "\n";
	

	// converting image to black and white
	int* bwBuffer = (int*)malloc(sizeof(int) * randomImage.height() * randomImage.width());
	makeImageBlackAndWhiteWrapper(randomImage.data(), randomImage.data() + (randomImage.height() * randomImage.width()), randomImage.data() + (2 * randomImage.height() * randomImage.width()), bwBuffer, randomImage.height(), randomImage.width());
	int* finalBuffer = (int*)malloc(sizeof(int) * 3* randomImage.height() * randomImage.width());
	// crop parts to fit neural net input size
	double testError = 0;
	int numBatchImages = 0;
	double* bwMatrix = (double*) malloc(sizeof(double) * numInputSquares * inputSize);
	double* colorMatrix = (double*) malloc(sizeof(double)*numInputSquares *outputSize);
	int* rowIndices = (int*)malloc(sizeof(int) * numInputSquares);
	int* colIndices = (int*)malloc(sizeof(int) * numInputSquares);

	cout << "running batch evaluate on all pixels for the image! ...\n";
	for (int i = 0;i < randomImage.height();i += outputSquareSide ) {
		for (int j = 0;j < randomImage.width();j += outputSquareSide ) {
			//cout << "on pixel i: " << i << "and pixel j: " << j << "\n";
			// getting the square for both the bw image and color image
			int* bwSquare = (int*)malloc(sizeof(int) * squareSide * squareSide);
			
			int* redSquare = (int*)malloc(sizeof(int) * outputSquareSide * outputSquareSide);
			int* greenSquare = (int*)malloc(sizeof(int) * outputSquareSide * outputSquareSide);
			int* blueSquare = (int*)malloc(sizeof(int) * outputSquareSide * outputSquareSide);
			

			/*
			cudaMemGetInfo(&freeMem, &totalMem);
			cout << "total memory of gpu: " << totalMem << "\n";
			cout << "total free memory of gpu before grabbing color squares: " << freeMem << "\n";
			*/

			// getting squares
			getSquareWrapper(bwBuffer, bwSquare, squareSide, randomImage.height(), randomImage.width(), i, j);
			getSquareWrapper(randomImage.data(), redSquare, outputSquareSide, randomImage.height(), randomImage.width(), i, j);
			getSquareWrapper(randomImage.data() + (randomImage.height() * randomImage.width()), greenSquare, outputSquareSide, randomImage.height(), randomImage.width(), i, j);
			getSquareWrapper(randomImage.data() + (2 * randomImage.height() * randomImage.width()), blueSquare, outputSquareSide, randomImage.height(), randomImage.width(), i, j);
			
			//cout << "got squares!\n";
			/*
			cudaMemGetInfo(&freeMem, &totalMem);
			cout << "total memory of gpu: " << totalMem << "\n";
			cout << "total free memory of gpu after grabbing color squares: " << freeMem << "\n";
			*/

			/*
			cudaMemGetInfo(&freeMem, &totalMem);
			cout << "total memory of gpu: " << totalMem << "\n";
			cout << "total free memory of gpu before grabbing bw square: " << freeMem << "\n";
			*/


			/*
			cudaMemGetInfo(&freeMem, &totalMem);
			cout << "total memory of gpu: " << totalMem << "\n";
			cout << "total free memory of gpu after grabbing bw square: " << freeMem << "\n";
			*/

			// scale pixels by 255 for both bw image and color image
			/*
			cudaMemGetInfo(&freeMem, &totalMem);
			cout << "total memory of gpu: " << totalMem << "\n";
			cout << "total free memory of gpu before scaling squares: " << freeMem << "\n";
			*/

			/*
			cudaMemGetInfo(&freeMem, &totalMem);
			cout << "total memory of gpu: " << totalMem << "\n";
			cout << "total free memory of gpu after scaling squares: " << freeMem << "\n";
			*/
			double* scaledBWSquare = (double*)malloc(sizeof(double) * squareSide * squareSide);
			pixelScaleWrapper(bwSquare, scaledBWSquare, squareSide, squareSide, inputPixelScaler);
			free(bwSquare);
			double* scaledRedSquare = (double*)malloc(sizeof(double) * outputSquareSide * outputSquareSide);
			pixelScaleWrapper(redSquare, scaledRedSquare, outputSquareSide, outputSquareSide,inputPixelScaler);
			free(redSquare);
			double* scaledGreenSquare = (double*)malloc(sizeof(double) * outputSquareSide * outputSquareSide);
			pixelScaleWrapper(greenSquare, scaledGreenSquare, outputSquareSide, outputSquareSide,inputPixelScaler);
			free(greenSquare);
			double* scaledBlueSquare = (double*)malloc(sizeof(double) * outputSquareSide * outputSquareSide);
			pixelScaleWrapper(blueSquare, scaledBlueSquare, outputSquareSide, outputSquareSide,inputPixelScaler);
			free(blueSquare);

			//cout << "scaled squares!\n";

			// copying memory to the accumulated input buffers so that we can evaluate the entire net on multiple inputs at once
			for (int z = 0;z < inputSize;z++) {
				bwMatrix[(z * numInputSquares) + numBatchImages] = scaledBWSquare[z];
			}

			for (int z = 0;z < outputSquareSide * outputSquareSide;z++) {
				colorMatrix[(z * numInputSquares) + numBatchImages] = scaledRedSquare[z];
				colorMatrix[((z+(outputSquareSide*outputSquareSide)) * numInputSquares) + numBatchImages] = scaledGreenSquare[z];
				colorMatrix[((z+(2*outputSquareSide*outputSquareSide)) * numInputSquares) + numBatchImages] = scaledBlueSquare[z];
			}

			rowIndices[numBatchImages] = i;
			colIndices[numBatchImages] = j;

			// freeing memory we no longer need
			free(scaledBWSquare);
			free(scaledRedSquare);
			free(scaledGreenSquare);
			free(scaledBlueSquare);

			numBatchImages++;

			//cout << "loaded batch Image: " << numBatchImages << "\n";

			if (numBatchImages != numInputSquares && (j+outputSquareSide < randomImage.width() || i+outputSquareSide < randomImage.height())) {
				continue;
			}

			if (numBatchImages!=numInputSquares) {
				//then we need to artificially fill the rest of the inputs, since we hit the end of input
				// and we should make sure that these do not contribute to our error calculations
				for (int z = numBatchImages;z < numInputSquares;z++) {
					for (int y = 0;y < inputSize;y++) {
						bwMatrix[(y * numInputSquares) + z] = 0;
					}
				}
			}
			// otherwise we have filled the input matrix to batch evaluate
			
			

			// evaluate net for each square vector in the input
			double* outputBuffer = (double*)malloc(sizeof(double) * numInputSquares * outputSize );
			/*
			cudaMemGetInfo(&freeMem, &totalMem);
			cout << "total free memory of gpu before evaluating net: " << freeMem << "\n";
			*/

			/*
			cudaMemGetInfo(&freeMem, &totalMem);
			cout << "total memory of gpu: " << totalMem << "\n";
			cout << "total free memory of gpu before evaluating net: " << freeMem << "\n";
			*/

			batchedGPUEvaluate(toTest, bwMatrix, outputBuffer);
			//cout << "finished batch evaluate for " << numBatchImages << " squares\n";

			//evaluateGPUNet(toTest, scaledBWSquare, outputBuffer);

			// error calculation 
			for (int z = 0;z < outputSquareSide * outputSquareSide;z++) {
				for (int k = 0;k < numInputSquares;k++) {
					if (k < numBatchImages) {
						testError += 0.5 * pow(outputBuffer[(z * numInputSquares) + k] - scaledRedSquare[z], 2);
						testError += 0.5 * pow(outputBuffer[(((outputSquareSide * outputSquareSide) + z) * numInputSquares) + k] - scaledGreenSquare[z], 2);
						testError += 0.5 * pow(outputBuffer[(((2 * outputSquareSide * outputSquareSide) + z) * numInputSquares) + k] - scaledBlueSquare[z], 2);
					}
				}
			}


			/*
			cudaMemGetInfo(&freeMem, &totalMem);
			cout << "total memory of gpu: " << totalMem << "\n";
			cout << "total free memory of gpu after evaluating net: " << freeMem << "\n";
			*/

			// iterating through vector output and setting final buffer data
			for (int k = 0;k < numBatchImages;k++) {
				int row = rowIndices[k];
				int col = colIndices[k];
				for (int z = 0;z < outputSquareSide*outputSquareSide;z++) {
					// error adjustment
					int rowAdjust = z / outputSquareSide;
					int colAdjust = z % outputSquareSide;
					int adjustedRow = row + rowAdjust;
					int adjustedCol = col + colAdjust;
					// all the reds are together, greens are together and blues are together
					if (adjustedRow < randomImage.height() && adjustedCol < randomImage.width()) {
						// then we put colors in their correct spot
						finalBuffer[((adjustedRow) * randomImage.width()) + (adjustedCol)] = outputBuffer[(z * numInputSquares)+k]*inputPixelScaler;
						finalBuffer[(randomImage.width() * randomImage.height())+((adjustedRow) * randomImage.width()) + (adjustedCol)] = outputBuffer[(((outputSquareSide*outputSquareSide)+z) * numInputSquares)+k]*inputPixelScaler;
						finalBuffer[(2*randomImage.width() * randomImage.height())+((adjustedRow) * randomImage.width()) + (adjustedCol)] = outputBuffer[(((2*outputSquareSide*outputSquareSide)+z) * numInputSquares)+k]*inputPixelScaler;	
					}
				
				}
			}
			

			/*
			cudaMemGetInfo(&freeMem, &totalMem);
			cout << "total memory of gpu: " << totalMem << "\n";
			cout << "total free memory of gpu before backprop net: " << freeMem << "\n";
			*/

			//resetting the number of batch images
			numBatchImages = 0;	
			free(outputBuffer);
		}
	}
	free(bwMatrix);
	free(colorMatrix);
	free(rowIndices);
	free(colIndices);
	

	// printing out the current training error
	printf("obtained test error: %lf\n", testError);

	/*/
	double* layer2Weights = (double*)malloc(sizeof(double) * toTrain->numInputs[1] * toTrain->numOutputs[1]);
	cudaMemcpy(layer2Weights, toTrain->weights[1], sizeof(double) * toTrain->numInputs[1] * toTrain->numOutputs[1], cudaMemcpyDeviceToHost);
	cout << "layer 2 weight 10: " << layer2Weights[10] << " \n";
	free(layer2Weights);
	*/

	CImg<int> testedOutput(finalBuffer, randomImage.width(), randomImage.height(), 1, 3);
		
	testedOutput.save("testWhileTrainOutput.png");

	//freeing allocated memory
	
	free(bwBuffer);
	free(finalBuffer);

	// returning the double error
	return testError;
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
		
		int trainCount = 0;
		// our loss function is (1/2)SUM(actualColor-predictedColor)^2
		double currTrainingError = (double)0;
		// buffers for batch training
		double* bwMatrix = (double*)malloc(sizeof(double) * inputSize * numInputSquares);
		double* colorMatrix = (double*)malloc(sizeof(double) * numInputSquares * outputSize);
		//int* rowIndices = (int*)malloc(sizeof(int) * inputSize);
		//int* colIndices = (int*)malloc(sizeof(int) * inputSize);
		int currBatches = 0;
		while (trainCount != epochNum*numInputSquares) {
			// pick a random image from the training dataset
			CImg<int> randomImage = getRandomTrainingImage();
			// convert image to black and white
			int* bwBuffer = (int*)malloc(sizeof(int) * randomImage.height() * randomImage.width());
			makeImageBlackAndWhiteWrapper(randomImage.data(), randomImage.data() + (randomImage.height() * randomImage.width()), randomImage.data() + (2 * randomImage.height() * randomImage.width()), bwBuffer, randomImage.height(), randomImage.width());
			// pick a random pixel from the image
			random_device rando;
			mt19937 gen(rando());
			uniform_int_distribution<> row(0, randomImage.height()-1);
			int randomPixelRow = row(gen);

			random_device rando2;
			mt19937 gen2(rando2());
			uniform_int_distribution<> col(0, randomImage.width()-1);
			int randomPixelCol = col(gen);

			// getting squares around the pixel for black and white and color
				int* bwSquare = (int*)malloc(sizeof(int) * squareSide * squareSide);

				int* redSquare = (int*)malloc(sizeof(int) * outputSquareSide * outputSquareSide);
				int* greenSquare = (int*)malloc(sizeof(int) * outputSquareSide * outputSquareSide);
				int* blueSquare = (int*)malloc(sizeof(int) * outputSquareSide * outputSquareSide);

				getSquareWrapper(bwBuffer, bwSquare, squareSide, randomImage.height(), randomImage.width(), randomPixelRow, randomPixelCol);
				getSquareWrapper(randomImage.data(), redSquare, outputSquareSide, randomImage.height(), randomImage.width(), randomPixelRow, randomPixelCol);
				getSquareWrapper(randomImage.data() + (randomImage.height() * randomImage.width()), greenSquare, outputSquareSide, randomImage.height(), randomImage.width(), randomPixelRow, randomPixelCol);
				getSquareWrapper(randomImage.data() + (2 * randomImage.height() * randomImage.width()), blueSquare, outputSquareSide, randomImage.height(), randomImage.width(), randomPixelRow, randomPixelCol);

				//debug
				/*
				for (int z = 0;z < outputSquareSide * outputSquareSide;z++) {
					cout << "red pixel value: " << redSquare[z] << "\n";
				} */


				// scaling squares by 255
				double* scaledBWSquare = (double*)malloc(sizeof(double) * squareSide * squareSide);
				pixelScaleWrapper(bwSquare, scaledBWSquare, squareSide, squareSide, inputPixelScaler);
				free(bwSquare);
				double* scaledRedSquare = (double*)malloc(sizeof(double) * outputSquareSide * outputSquareSide);
				pixelScaleWrapper(redSquare, scaledRedSquare, outputSquareSide, outputSquareSide, inputPixelScaler);
				free(redSquare);
				double* scaledGreenSquare = (double*)malloc(sizeof(double) * outputSquareSide * outputSquareSide);
				pixelScaleWrapper(greenSquare, scaledGreenSquare, outputSquareSide, outputSquareSide, inputPixelScaler);
				free(greenSquare);
				double* scaledBlueSquare = (double*)malloc(sizeof(double) * outputSquareSide * outputSquareSide);
				pixelScaleWrapper(blueSquare, scaledBlueSquare, outputSquareSide, outputSquareSide, inputPixelScaler);
				free(blueSquare);

				// filling the bwmatrix and color matrix for batched training
				for (int z = 0;z < inputSize;z++) {
					bwMatrix[(z * numInputSquares) + currBatches] = scaledBWSquare[z];
				}
				//double* accumulatedActualValues = (double*)malloc(sizeof(double) * outputSize);

				for (int z = 0;z < outputSquareSide * outputSquareSide;z++) {
					/*
					accumulatedActualValues[z] = scaledRedSquare[z];
					accumulatedActualValues[(outputSquareSide * outputSquareSide) + z] = scaledGreenSquare[z];
					accumulatedActualValues[(2*outputSquareSide * outputSquareSide) + z] = scaledBlueSquare[z];
					*/

					colorMatrix[(z * numInputSquares) + currBatches] = scaledRedSquare[z];
					colorMatrix[((z + (outputSquareSide * outputSquareSide)) * numInputSquares) + currBatches] = scaledGreenSquare[z];
					colorMatrix[((z + (2 * outputSquareSide * outputSquareSide)) * numInputSquares) + currBatches] = scaledBlueSquare[z];

				}

				//rowIndices[currBatches] = i;
				//colIndices[currBatches] = j;

				// freeing unnecessary data 

				free(bwBuffer);
				free(scaledBWSquare);
				free(scaledRedSquare);
				free(scaledGreenSquare);
				free(scaledBlueSquare);

				// checking how many batches we have 
				currBatches++;
				trainCount++;
				cout << "FINISHED EPOCH: " << trainCount << "\n";
				if (currBatches != numInputSquares) {
					// then we grab more squares
					continue;
				}

				//resetting curr batches
				currBatches = 0;

				// running output of the scaled black and white squares

				//while (true) {
					double* outputBuffer = (double*)malloc(sizeof(double) * outputSize * numInputSquares);

					int count = 0;
					//evaluateGPUNet(toTrain, scaledBWSquare, outputBuffer);
					batchedGPUEvaluate(toTrain, bwMatrix, outputBuffer);


					// incrementing training error	
					/*
					for (int z = 0;z < outputSize;z++) {
						currTrainingError += 0.5 * pow(outputBuffer[z] - accumulatedActualValues[z], 2);
					} */
					for (int z = 0;z < outputSize * numInputSquares;z++) {
						currTrainingError += 0.5 * pow(outputBuffer[z] - colorMatrix[z], 2);
						//debug
						/*
						cout << "obtained pixel value: " << outputBuffer[z] * inputPixelScaler << "\n";
						cout << "actual pixel value: " << colorMatrix[z] * inputPixelScaler << "\n";
						*/

					}
					//debug
					/*
					cout << "current error: " << currTrainingError << "\n";
					currTrainingError = 0;
					*/


					// backprop for the batch
					batchBackPropogation(toTrain, colorMatrix, outputBuffer, learningRate);

					//debugging for output	
					/*
					for (int z = 0;z < outputSize;z++) {
						outputBuffer[z] *= inputPixelScaler;
					}


					if (count == 0) {
						for (int z = 0;z < outputSize;z++) {
							accumulatedActualValues[z] *= inputPixelScaler;
						}
						CImg<double> actualColor(accumulatedActualValues, outputSquareSide, outputSquareSide, 1, 3);
						actualColor.save("actualPatch.jpg", trainCount);
						for (int z = 0;z < outputSize;z++) {
							accumulatedActualValues[z] /= inputPixelScaler;
						}
						count++;
					}

					CImg<double> guessedColor(outputBuffer, outputSquareSide, outputSquareSide, 1, 3);
					guessedColor.save("guessedPatch.jpg", trainCount);
					*/

					// freeing memory we no longer need
					/*
					free(accumulatedActualValues);
					free(scaledBWSquare);
					free(scaledRedSquare);
					free(scaledGreenSquare);
					free(scaledBlueSquare);
					free(bwBuffer);
					*/
					free(outputBuffer);
				//}


				// printing out the current training error
				printf("current training error: %lf\n", currTrainingError);
		}
		free(bwMatrix);
		free(colorMatrix);
		//free(rowIndices);
		//free(colIndices);
		// then we will perform a test of error on a random test data image
		// writing weights back to filesystem now that the epochLimit was reached
		cout << "writing updated weights to filesystem\n";
		writeGPUNet(toTrain);
		cout << "testing net on testing data now...\n";
		// forget testing for now, since we need to train a lot more
		double testError = testFromTestData(toTrain);
	}	
}

