#ifndef MAIN_H
#define MAIN_H

#include <cuda_runtime.h>
#include <unistd.h>
#include <iostream>
#include <string>

#define ERR(cond, msg)\
	if (cond) {\
		std::cerr << "error: " msg << std::endl;\
		_exit(-1);\
	}

#define CUDAERR(fn, msg)\
	if ((err = fn) != cudaSuccess) {\
		std::cerr << "cuda error: " msg " ("\
			<< cudaGetErrorString(err) << ")" << std::endl;\
		_exit(-1);\
	}

typedef unsigned char byte;

// image, filter, and cuda properties
extern cudaError_t err;
extern float *dImg, *dTmp1, *dTmp2;	// dTmp* used for intermediate outputs
extern unsigned rowStride, channels, bufSize, blockSize;
extern dim3 dimGrid, dimBlock;

// defined in pngio.c
extern int height, width;

// util.cu
__host__ void alloc_copy_htd(void *hptr, void **dptr, unsigned size,
	std::string name);
__host__ void copy_dth(void *hptr, void *dptr, unsigned size,
	std::string name);
__host__ void free_d(void *dptr, std::string name);

// blur.cu
__host__ void gaussian_filter(float blurStd, float **fltp, unsigned *fltSizep);
__host__ void blur(int blurSize);

// deblur.cu
__host__ void deblur(void);

// conv2d.cu
__global__ void conv2d(float *d1, float *d2, float *d3, int ch,
	int h1, int w1, int h2, int w2);

#endif // MAIN_H
