// including the image processing library and jpg processing support
#include <stdio.h>
#include "CImg.h"

// including our cuda wrapper functions
#include "imageKernel.cuh"
#include <malloc.h>

// adding our neural net logic
#include "NeuralNet.h"

#include "GPUSecondNeuralNet.cuh"
#include <iostream>

using namespace cimg_library;
using namespace std;

/*Function Prototypes*/
// below are gpu calls
void GPUBlackAndWhiteCreation(char* colorImageName, char* outputName);
void GPUConvertColorImageTo4K(char* colorImageName, char* outputName);
void GPUConvertBWImageTo4k(char* blackWhiteName, char* outputName);
void GPUShowImageFromPatch(char* blackWhiteImage, char* outputPatchName);

// below are cpu calls (in case you are having trouble getting gpu calls to work out)
void CPUBlackAndWhiteCreation(char* colorImageName, char* outputName);
void CPUConvertColorImageTo4K(char* colorImageName, char* outputName);
void CPUConvertBWImageTo4k(char* blackWhiteName, char* outputName);
void CPUShowImageFromPatch(char* blackWhiteImage, char* outputPatchName);

int main(int argc, char* argv[]) {
	// testing if we can load the testData jpg and then print out stuff about it using the CImg library and imagemagick
	
	//GPUBlackAndWhiteCreation("TrainingData/lordOfRings.jpg", "lotrBW.png");
	//GPUConvertColorImageTo4K("testData.jpg", "colorConverted4k.jpg");
	//GPUConvertBWImageTo4k("blackAndWhite.jpg", "blackWhiteConverted4k.jpg");
	//GPUShowImageFromPatch("blackWhiteConverted4k.jpg", "testPatchOutput.jpg");
	
	//outputFromNeuralNet("TrainingData/animal-6563620_1920.jpg", "hopeThisWorks.jpg");
	/* 
	//TESTING GPU INITIALIZATION AND WRITING
	GPUNet* initialized = loadGPUNet();
	writeGPUNet(initialized);
	*/
	/*
	//TESTING SIGMOID GPU KERNEL
	double* input = (double*)malloc(sizeof(double) * 1000);
	double* output = (double*)malloc(sizeof(double) * 1000);
	//randomizing input
	for (int i = 0;i < 1000;i++) {
		srand(time(NULL));
		input[i] = (double) (rand() / (RAND_MAX * 4.0));
	}
	sigmoidMatrixTest(input, output, 1000);
	*/

	//CImg<double> erai("TestData/boat-6561172_1920.png");
	//getSquaresTest(erai.data(), erai.height(), erai.width(), 300000, 31);

	//testFromTestData(NULL);
	
	trainFromDataSet(0.6);
	//outputFromGPUNet("boatBW.jpg", "Crazyoutput.png");
	//if(testCublas()) cout << "SUCCESSS!\n";
	
	/*
	CImg<double> test("TestData/brooklynNineNineTest.jfif");
	test.RGBtoLab();
	cout << "spectrum: " << test.spectrum() << "\n";
	cout << "width: " << test.width() << "\n";
	cout << "height: " << test.width() << "\n";
	cout << "depth: " << test.depth() << "\n";
	*/

	/*
	for (int i = 0;i < test.height();i++) {
		for (int j = 0;j < test.width();j++) {
			cout << "Luminance: " << test.get_shared_channel(0)[(i*test.width())+j] << "\n";
		}
	} */
	/*
	CImg<double> converted(test.get_shared_channel(0),test.width(),test.height());
	converted.save("output.jpg");
	*/
	
	
	
	
	
	return 0;
}


