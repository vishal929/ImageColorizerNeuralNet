#ifndef imagekernel_h
#define imagekernel_h

void makeImageBlackAndWhiteWrapper(int* colorR, int* colorG, int* colorB, int* bwImage, int rowDim, int colDim);
void makeColorImage4KWrapper(int* colorR, int* colorG, int* colorB, int* newR, int* newG, int* newB, int rowDim, int colDim);

#endif
