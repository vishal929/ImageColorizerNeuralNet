#ifndef imagekernel_h
#define imagekernel_h

#define defaultPatchSize 301

void makeImageBlackAndWhiteWrapper(int* colorR, int* colorG, int* colorB, int* bwImage, int rowDim, int colDim);
void makeColorImage4kWrapper(int* colorR, int* colorG, int* colorB, int* newR, int* newG, int* newB, int rowDim, int colDim);
void makeBlackWhiteImage4KWrapper(int* bwValues, int* newBWValues, int rowDim, int colDim);
void getPatchWrapper(double* imagePixels, double* imagePatch, int rowDim, int colDim, int patchSize, int features, int pixelRow, int pixelCol);

#endif
