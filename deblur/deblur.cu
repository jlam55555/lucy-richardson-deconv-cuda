#include <iostream>
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
 * 	0 <= n-i < w2 => w2-n < i <= n
 * 2d convolution: (x*y)[n,m] = sum_i {sum_j {x[i,j]y[x-i,y-j]}}
 */

// performs a 2d convolution; d3 should be the same size as d1;
// assumes that d1's dimensions > d2's dimensions
__global__ void conv2d(float *d1, float *d2, float *d3, int ch,
	int h1, int w1, int h2, int w2)
{
	unsigned int y, x, c, i, j, sum, imin, imax, jmin, jmax, rs;

	// infer y, x, c from block/thread index
	y = blockDim.y * blockIdx.y + threadIdx.y;
	x = blockDim.x * blockIdx.x + threadIdx.x;
	c = x % 3;
	x /= 3;

	// out of bounds, no work to do
	if (x >= w1 || y >= h1) {
		return;
	}

	// convolution result
	sum = 0;

	// appropriate ranges for convolution
	imin = min(max(0, h2-y+1), h1);
	imax = min(h1, y);
	jmin = min(max(0, w2-x+1), w1);
	jmax = min(w1, x);

	// row stride (width * number of channels)
	rs = ch*w1;

	// convolution
	for (i = imin; i < imax; ++i) {
		for (j = jmin; j < jmax; ++j) {
			sum += d1[i*rs + j*c + c] * d2[(x-i)*rs + (y-j)*c + c];
		}
	}

	// set result
	d3[x*rs + y*c + c] = sum;
}

// copy d1 to d2, but change from unsigned char to float
__global__ void byteToFloat(byte *d1, float *d2, int h, int rs)
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
__global__ void floatToByte(float *d1, byte *d2, int h, int rs)
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
__global__ void invert(float *d1, int h, int rs, int isAlpha)
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

// image and cuda properties
cudaError_t err = cudaSuccess;
float *dImg = nullptr;
unsigned int rowStride, channels, bufSize, blockSize;
dim3 dimGrid, dimBlock;

// image processing routines go here
__host__ int processImage(void)
{
	// invert image (for testing)
	invert<<<dimGrid, dimBlock>>>(dImg, height, rowStride,
		color_type==PNG_COLOR_TYPE_RGBA);
	CUDAERR(cudaGetLastError(), "launch invert kernel");

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
	CUDAERR(cudaFree(dImgPix), "freeing dImgPix");
	free(hImgPix);

	// write file
	std::cout << "Writing file..." << std::endl;
	write_png_file("out.png");

	std::cout << "Done." << std::endl;
	return 0;
}
