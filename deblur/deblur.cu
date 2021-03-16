#include <iostream>
#include <cuda_runtime.h>
#include "pngio.h"

typedef unsigned char byte;

#define CUDAERR(fn, msg)\
	if ((err = fn) != cudaSuccess) {\
		fprintf(stderr, "error: " #msg " (%s)\n",\
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
	if (x > w1 || y > h1) {
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
__global__ void byte_to_float(byte *d1, float *d2, int h, int rs)
{
	unsigned int y, x;

	// infer y, x, c from block/thread index
	y = blockDim.y * blockIdx.y + threadIdx.y;
	x = blockDim.x * blockIdx.x + threadIdx.x;

	d2[y*rs + x] = d1[y*rs + x];
}

// copy d1 to d2, but change from float to unsigned char
__global__ void float_to_byte(byte *d1, float *d2, int h, int rs)
{
	unsigned int y, x;

	// infer y, x, c from block/thread index
	y = blockDim.y * blockIdx.y + threadIdx.y;
	x = blockDim.x * blockIdx.x + threadIdx.x;

	d2[y*rs + x] = d1[y*rs + x];
}

__host__ int main(void)
{
	// allocate buffers for image, copy into contiguous array
	byte *hImgPix = nullptr, *dImgPix = nullptr;
	float *dImg = nullptr;
	cudaError_t err = cudaSuccess;
	unsigned int rowStride, channels, bufSize, y;

	// read input file
	std::cout << "Reading file..." << std::endl;
	read_png_file("test.png");

	// assuming RGBA
	// TODO: don't assume RGBA, get from image
	channels = 4;
	rowStride = width * channels;
	bufSize = rowStride * height;

	// allocate host buffer, copy image to buffers
	hImgPix = (byte *) malloc(bufSize*sizeof(byte));
	if (!hImgPix) {
		fprintf(stderr, "Error allocating contiguous buffer for "
			"image\n");
		return -1;
	}

	// allocate other buffers
	CUDAERR(cudaMalloc((void **) &dImgPix, bufSize), "allocating dImgPix");
//	err = cudaMalloc((void **) &dImgPix, bufSize);
//	if (err != cudaSuccess) {
//		fprintf(stderr, "Error allocating dImgPix\n");
//		return -1;
//	}
	CUDAERR(cudaMalloc((void **) &dImg, bufSize*sizeof(float)),
		"allocating dImg");
//	err = cudaMalloc((void **) &dImg, bufSize*sizeof(float));
//	if (err != cudaSuccess) {
//		fprintf(stderr, "Error allocating dImg\n");
//		return -1;
//	}

	// copy image to device (hImgPix -> dImgPix)
	for (y = 0; y < height; ++y) {
		CUDAERR(cudaMemcpy(dImgPix+rowStride*y, hImgPix+rowStride*y,
			rowStride, cudaMemcpyHostToDevice),
			"copying image to device");
//		err = cudaMemcpy(dImgPix+rowStride*y, hImgPix+rowStride*y,
//			rowStride, cudaMemcpyHostToDevice);
//		if (err != cudaSuccess) {
//			fprintf(stderr, "error copying image to device\n");
//			return -1;
//		}
	}

	// convert image to float (dImgPix -> dImg)

	// create gaussian filter

	// apply gaussian filter

	// convert image back to byte (dImg -> dImgPix)

	// copy image back (dImgPix -> hImgPix)
	for (y = 0; y < height; ++y) {
		CUDAERR(cudaMemcpy(hImgPix+rowStride*y, dImgPix+rowStride*y,
			rowStride, cudaMemcpyDeviceToHost),
			"copying image to host");
//		err = cudaMemcpy(hImgPix+rowStride*y, dImgPix+rowStride*y,
//				 rowStride, cudaMemcpyDeviceToHost);
//		if (err != cudaSuccess) {
//			fprintf(stderr, "error copying image to host\n");
//			return -1;
//		}
	}

	// free buffers
	CUDAERR(cudaFree(dImg), "freeing dImg");
//	err = cudaFree(dImg);
//	if (err != cudaSuccess) {
//		fprintf(stderr, "error freeing dImg\n");
//		return -1;
//	}
	CUDAERR(cudaFree(dImgPix), "freeing dImgPix");
//	err = cudaFree(dImgPix);
//	if (err != cudaSuccess) {
//		fprintf(stderr, "error freeing dImgPix\n")
//		return -1;
//	}
	free(hImgPix);

	std::cout << "Writing file..." << std::endl;
	write_png_file("out.png");

	std::cout << "Done." << std::endl;
	return 0;
}
