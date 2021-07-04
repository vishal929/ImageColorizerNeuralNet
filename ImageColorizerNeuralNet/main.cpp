// including the image processing library
#include "./CImg_latest/CImg-2.9.8_pre061621/CImg.h"

// including our cuda wrapper functions
#include "imageKernel.cuh"

using namespace cimg_library;

/*Function Prototypes*/
void blackAndWhiteCreation(char* colorImageName, char* outputName);

int main(int argc, char* argv[]) {
	// testing if we can load the testData jpg and then print out stuff about it using the CImg library and imagemagick
	blackAndWhiteCreation("testData.jpg", "blackAndWhite.jpg");
	return 0;
}

// helper function to just take in an image and output a black and white version using the wrapped cuda kernel call
void blackAndWhiteCreation(char* colorImageName, char* outputName) {
	CImg<int> testImage(colorImageName);
	// seeing if we can call the cuda method to convert this image to black and white, and then output the black and white image
	// getting pointers to RGB sections and passing this to the cuda wrapper
	int* colorR = testImage.data();
	int* colorG = colorR + (testImage.height() * testImage.width());
	int* colorB = colorG + (testImage.height() * testImage.width());
	int* bwBuffer = (int*) malloc(sizeof(int) * testImage.width() * testImage.height());
	// calling the cuda wrapper
	makeImageBlackAndWhiteWrapper(colorR, colorG, colorB, bwBuffer, testImage.height(), testImage.width());
	int* finalBWBuffer = (int*)malloc(sizeof(int) * testImage.width() * testImage.height() * 3);
	int otherCount = 0;
	for (int i = 0; i < testImage.width() * testImage.height(); i++) {
		finalBWBuffer[otherCount] = bwBuffer[i];
		finalBWBuffer[otherCount + 1] = bwBuffer[i];
		finalBWBuffer[otherCount + 2] = bwBuffer[i];
		otherCount += 3;
	}
	CImg<int> bwImage(bwBuffer, testImage.width(), testImage.height());
	bwImage.save(outputName);
}