#ifndef cudaErrorHandler_h
#define cudaErrorhandler_h


#include <stdio.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>

#define cudaErrorCheck(error) {detailError((error), __FILE__, __LINE__);}

inline void detailError(cudaError_t error, const char* file, int line) {
	if (error != cudaSuccess) {
		fprintf(stderr, "error in file:%s and line:%d --> error: %s", file, line, cudaGetErrorString(error));
		exit(error);
	}
}
#endif