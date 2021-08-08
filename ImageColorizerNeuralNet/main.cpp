// including the image processing library and jpg processing support
#include <stdio.h>
#include "CImg.h"

// including our cuda wrapper functions
#include "imageKernel.cuh"
#include <malloc.h>

using namespace cimg_library;

/*Function Prototypes*/
void blackAndWhiteCreation(char* colorImageName, char* outputName);
void convertColorImageTo4K(char* colorImageName, char* outputName);

int main(int argc, char* argv[]) {
	// testing if we can load the testData jpg and then print out stuff about it using the CImg library and imagemagick
	blackAndWhiteCreation("testData.jpg", "blackAndWhite.jpg");
	convertColorImageTo4K("testData.jpg", "colorConverted4k.jpg");
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
	/*
	int* finalBWBuffer = (int*)malloc(sizeof(int) * testImage.width() * testImage.height() * 3);
	int otherCount = 0;
	for (int i = 0; i < testImage.width() * testImage.height(); i++) {
		finalBWBuffer[otherCount] = bwBuffer[i];
		finalBWBuffer[otherCount + 1] = bwBuffer[i];
		finalBWBuffer[otherCount + 2] = bwBuffer[i];
		otherCount += 3;
	} */
	CImg<int> bwImage(bwBuffer, testImage.width(), testImage.height());
	bwImage.save(outputName);
}

// idea behind this is to take an image <= to 4k resolution, and expand it if needed with black pixels to 4k resolution. This is to standardize our images for training/evaluating based on the model (for color images)
// returns a pointer to the converted image
// todo: possibly make use of pthreads to speed this up? 
void convertColorImageTo4K(char* colorImageName, char* outputName) {
	// (4k is 3480 x 2160)
	// black pixel value is 0 for red, green, and blue	
	// allocating memory for our new image data
	CImg<int> testImage(colorImageName);
	int* colorR = testImage.data();
	int* colorG = colorR + (testImage.height() * testImage.width());
	int* colorB = colorG + (testImage.height() * testImage.width());
	const int targetRes = 3480 * 2160;
	int* newR = (int*) malloc(sizeof(int)*targetRes);
	int* newG = (int*) malloc(sizeof(int)*targetRes);
	int* newB = (int*) malloc(sizeof(int)*targetRes);
	printf("width is %d and height is %d\n", testImage.width(), testImage.height());
	for (int i = 0; i < 2160; i++) {
		for (int j = 0; j < 3480; j++) {
			// printf("i value: %d --> j value: %d \n", i, j);
			if (i >= testImage.height() || j >= testImage.width()) {
				// then this pixel will be black for all channels
				newR[(i * 3480) + j] = 0;
				newG[(i * 3480) + j] = 0;
				newB[(i * 3480) + j] = 0;
			}
			else {
				// then this pixel is just the same as the original
				newR[(i * 3480) + j] = colorR[(i*testImage.width())+j];
				newG[(i * 3480) + j] = colorG[(i*testImage.width())+j];
				newB[(i * 3480) + j] = colorB[(i*testImage.width())+j];
			}
		}
	}

	// creating the new image
	CImg<int> scaledColorImage(3840,2160,1,3);
	// filling the image
	for (int i = 0; i < targetRes; i++) {
		scaledColorImage.fill(newR[i]);
	}
	for (int i = 0; i < targetRes; i++) {
		scaledColorImage.fill(newG[i]);
	}
	for (int i = 0; i < targetRes; i++) {
		scaledColorImage.fill(newB[i]);
	}
	scaledColorImage.save(outputName);
	free(newR);
	free(newG);
	free(newB);

}

// (same as above but for black and white images
void convertBWImageTo4k() {

}