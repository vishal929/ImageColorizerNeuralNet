#ifndef imagekernel_h
#define imagekernel_h

void makeImageBlackAndWhiteWrapper(int* colorR, int* colorG, int* colorB, int* bwImage, int rowDim, int colDim);
void makeColorImage4kWrapper(int* colorR, int* colorG, int* colorB, int* newR, int* newG, int* newB, int rowDim, int colDim);
void makeBlackWhiteImage4KWrapper(int* bwValues, int* newBWValues, int rowDim, int colDim);

#endif
