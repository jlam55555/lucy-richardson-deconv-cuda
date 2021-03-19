#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "pngio.h"

typedef unsigned char byte;

#define ERR(cond, msg)\
	if (cond) {\
		fprintf(stderr, "error: " msg "\n");\
		return -1;\
	}

#define CUDAERR(fn, msg)\
	if ((err = fn) != cudaSuccess) {\
		fprintf(stderr, "cuda error: " msg " (%s)\n",\
			cudaGetErrorString(err));\
		return -1;\
	}

/*
 * Review:
 * 1d convolution: (x*y)[n] = sum_i {x[i]y[n-i]}
 * 	0 <= i < w1
 * 	0 <= n-i < w2 => n-w2 < i <= n
 *		=> max(0, n-w2+1) <= i < min(w1, n+1)
 * 2d convolution: (x*y)[n,m] = sum_i {sum_j {x[i,j]y[x-i,y-j]}}
 */

// performs a 2d convolution d3=d1*d2; d3 should be the same size as d1;
// assumes that d1's dimensions > d2's dimensions; this treats d2 like a
// filter, and applies it centered at each point of d1
__global__ void static conv2d(float *d1, float *d2, float *d3, int ch,
	int h1, int w1, int h2, int w2)
{
	unsigned int y, x, c, i, j, imin, imax, jmin, jmax, rs, ip, jp;
	float sum;

	// infer y, x, c from block/thread index
	y = blockDim.y * blockIdx.y + threadIdx.y;
	x = blockDim.x * blockIdx.x + threadIdx.x;
	c = x % ch;
	x /= ch;

	// out of bounds, no work to do
	if (x >= w1 || y >= h1) {
		return;
	}

	// appropriate ranges for convolution
	imin = max(0, y+h2/2-h2+1);
	imax = min(h1, y+h2/2+1);
	jmin = max(0, x+w2/2-w2+1);
	jmax = min(w1, x+w2/2+1);

	// row stride (width * number of channels)
	rs = ch*w1;

	// convolution
	// TODO: this only deals with the case where d2 has a single channel
	//	(i.e., like a filter)
	sum = 0;
	for (i = imin; i < imax; ++i) {
		for (j = jmin; j < jmax; ++j) {
			ip = i - h2/2;
			jp = j - w2/2;

			sum += d1[i*rs + j*ch + c] * d2[(y-ip)*w2 + (x-jp)];
		}
	}

	// set result
	d3[y*rs + x*ch + c] = sum;
}

// copy d1 to d2, but change from unsigned char to float
__global__ static void byteToFloat(byte *d1, float *d2, int h, int rs)
{
	unsigned int y, x;

	// infer y, x, c from block/thread index
	y = blockDim.y * blockIdx.y + threadIdx.y;
	x = blockDim.x * blockIdx.x + threadIdx.x;
	if (y >= h || x >= rs) {
		return;
	}

	d2[y*rs + x] = d1[y*rs + x];
}

// copy d1 to d2, but change from float to unsigned char
__global__ static void floatToByte(float *d1, byte *d2, int h, int rs)
{
	unsigned int y, x;

	// infer y, x, c from block/thread index
	y = blockDim.y * blockIdx.y + threadIdx.y;
	x = blockDim.x * blockIdx.x + threadIdx.x;
	if (y >= h || x >= rs) {
		return;
	}

	d2[y*rs + x] = d1[y*rs + x];
}

// simple filter for testing purposes: invert colors
__global__ static void invert(float *d1, int h, int rs, int isAlpha)
{
	unsigned int y, x;

	// infer y, x, c from block/thread index
	y = blockDim.y * blockIdx.y + threadIdx.y;
	x = blockDim.x * blockIdx.x + threadIdx.x;

	// x%4==3: don't invert alpha channel if applicable
	if (y >= h || x >= rs || (isAlpha && x%4==3)) {
		return;
	}

	d1[y*rs + x] = 255-d1[y*rs + x];
}

// image, filter, and cuda properties
static cudaError_t err = cudaSuccess;
static float *dImg, *dTmp;	// dTmp used for intermediate outputs
static unsigned int rowStride, channels, bufSize, blockSize;
static dim3 dimGrid, dimBlock;

// image processing routines go here
__host__ static int processImage(void)
{
	float *hFlt, *dFlt, fltSum, blurStd, cent;
	unsigned int i, j, fltSize;

/*
	// invert image (for testing)
	invert<<<dimGrid, dimBlock>>>(dImg, height, rowStride,
		color_type==PNG_COLOR_TYPE_RGBA);
	CUDAERR(cudaGetLastError(), "launch invert kernel");
*/

	// initialize filter; 3x3 circular gaussian filter
	// https://en.wikipedia.org/wiki/Gaussian_blur
	blurStd = 2;
	fltSize = 6*blurStd+1;	// for factor of 6 see Wikipedia
				// +1 to make it odd for better centering
	cent = (fltSize-1.)/2;	// center of filter

	ERR(!(hFlt = (float *) malloc(fltSize*fltSize*sizeof(float))),
		"allocate hFlt");
	fltSum = 0;
	for (i = 0; i < fltSize; ++i) {
		for (j = 0; j < fltSize; ++j) {
			hFlt[i*fltSize+j] = exp(-(pow(i-cent, 2)+pow(j-cent, 2))
				/(2*blurStd*blurStd))/(2*M_PI*blurStd*blurStd);
			fltSum += hFlt[i*fltSize+j];
		}
	}

	// normalize the filter
	for (i = 0; i < fltSize*fltSize; ++i) {
		hFlt[i] /= fltSum;
	}

	// allocate and copy filter to device
	CUDAERR(cudaMalloc((void **) &dFlt, fltSize*fltSize*sizeof(float)),
		"allocating dFlt");
	CUDAERR(cudaMemcpy(dFlt, hFlt, fltSize*fltSize*sizeof(float),
		cudaMemcpyHostToDevice), "copying hFlt to device");

	// blur image (for testing)
	conv2d<<<dimGrid, dimBlock>>>(dImg, dFlt, dTmp, channels,
		height, width, fltSize, fltSize);

	// result is currently in dTmp, move to dImg
	CUDAERR(cudaMemcpy(dImg, dTmp, bufSize*sizeof(float),
		cudaMemcpyDeviceToDevice), "copying dTmp to dImg");

	// cleanup
	free(hFlt);
	CUDAERR(cudaFree(dFlt), "freeing dFlt");
	return 0;
}

// driver for function
__host__ int main(int argc, char **argv)
{
	// allocate buffers for image, copy into contiguous array
	byte *hImgPix = nullptr, *dImgPix = nullptr;
	unsigned int y;

	// get input file from stdin
	ERR(argc < 2, "missing input file as cmd parameter\n"
		"\tusage: ./deblur [INPUT_FILE].png");

	// read input file
	std::cout << "Reading file..." << std::endl;
	read_png_file(argv[1]);

	// assume only RGB (3 channels) or RGBA (4 channels)
	channels = color_type==PNG_COLOR_TYPE_RGBA ? 4 : 3;
	rowStride = width * channels;
	bufSize = rowStride * height;

	// allocate host buffer, copy image to buffers
	ERR(!(hImgPix = (byte *) malloc(bufSize)),
		"allocating contiguous buffer for image");

	// allocate other buffers
	CUDAERR(cudaMalloc((void **) &dImgPix, bufSize), "allocating dImgPix");
	CUDAERR(cudaMalloc((void **) &dImg, bufSize*sizeof(float)),
		"allocating dImg");
	CUDAERR(cudaMalloc((void **) &dTmp, bufSize*sizeof(float)),
		"allocating dTmp");

	// copy image to contiguous buffer (double pointer is not guaranteed
	// to be contiguous)
	for (y = 0; y < height; ++y) {
		memcpy(hImgPix+rowStride*y, row_pointers[y], rowStride);
	}

	// copy image to device (hImgPix -> dImgPix)
	CUDAERR(cudaMemcpy(dImgPix, hImgPix, bufSize, cudaMemcpyHostToDevice),
		"copying image to device");

	// set kernel parameters (same for all future kernel invocations)
	blockSize = 32;
	dimGrid = dim3(ceil(rowStride*1./blockSize),
		ceil(height*1./blockSize), 1);
	dimBlock = dim3(blockSize, blockSize, 1);

	// convert image to float (dImgPix -> dImg)
	byteToFloat<<<dimGrid, dimBlock>>>(dImgPix, dImg, height, rowStride);
	CUDAERR(cudaGetLastError(), "launch byteToFloat kernel");

	// image processing routine
	std::cout << "Processing image..." << std::endl;
	if (processImage() < 0) {
		return -1;
	}

	// convert image back to byte (dImg -> dImgPix)
	floatToByte<<<dimGrid, dimBlock>>>(dImg, dImgPix, height, rowStride);
	CUDAERR(cudaGetLastError(), "launch floatToByte kernel");

	// copy image back (dImgPix -> hImgPix)
	CUDAERR(cudaMemcpy(hImgPix, dImgPix, bufSize, cudaMemcpyDeviceToHost),
		"copying image from device");

	// copy image back into original pixel buffers
	for (y = 0; y < height; ++y) {
		memcpy(row_pointers[y], hImgPix+rowStride*y, rowStride);
	}

	// free buffers
	CUDAERR(cudaFree(dImg), "freeing dImg");
	CUDAERR(cudaFree(dTmp), "freeing dTmp");
	CUDAERR(cudaFree(dImgPix), "freeing dImgPix");
	free(hImgPix);

	// write file
	std::cout << "Writing file..." << std::endl;
	write_png_file("out.png");

	std::cout << "Done." << std::endl;
	return 0;
}
