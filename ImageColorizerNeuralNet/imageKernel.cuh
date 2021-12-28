#ifndef imagekernel_h
#define imagekernel_h

#define defaultPatchSize 301

void makeImageBlackAndWhiteWrapper(double* colorR, double* colorG, double* colorB, double* bwImage, int rowDim, int colDim);
void makeColorImage4kWrapper(int* colorR, int* colorG, int* colorB, int* newR, int* newG, int* newB, int rowDim, int colDim);
void makeBlackWhiteImage4KWrapper(int* bwValues, int* newBWValues, int rowDim, int colDim);
void getPatchWrapper(double* imagePixels, double* imagePatch, int rowDim, int colDim, int patchSize, int features, int pixelRow, int pixelCol);
void pixelScaleWrapper(double* inputPixels, double* outputValues, int rowDim, int colDim, double scalar);
void getSquareWrapper(double* inputPixels, double* squarePixels, int squareSideLength, int rowDim, int colDim, int pixelRow, int pixelCol);
void getSquareCPU(double* inputPixels, double* squarePixels, int squareSideLength, int rowDim, int colDim, int pixelRow, int pixelCol);
void getSquaresWrapper(double* inputPixels, double* squares, int squareSideLength, int rowDim, int colDim, int numSquares, int startPixel);
void getSquaresTest(double* data, int rowDim, int colDim, int numSquaresToCheck, int squareLength);

#endif
