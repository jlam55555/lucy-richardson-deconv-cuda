#include <cmath>
#include "main.h"
#include "pngio.h"

// these are declared in main.h
cudaError_t err = cudaSuccess;
float *dImg, *dTmp1, *dTmp2, *dTmp3;
unsigned int rowStride, channels, bufSize, blockSize;
dim3 dimGrid, dimBlock;

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
	if (y >= h || x >= rs || (x%4==3)) {
		return;
	}

	d2[y*rs + x] = min(max(d1[y*rs + x], 0.), 255.);
}

// image processing routines go here
__host__ static void processImage(void)
{
	// perform 25 iterations of the LR deconvolution
	// with a gaussian kernel with sigma=2
	deblur(25, 2);
}

// driver for function
__host__ int main(int argc, char **argv)
{
	// allocate buffers for image, copy into contiguous array
	byte *hImgPix = nullptr, *dImgPix = nullptr;
	clock_t *t;
	unsigned int y;

	// get input file from stdin
	ERR(argc < 3, "usage: ./deblur INPUT.png OUTPUT.png");

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
	CUDAERR(cudaMalloc((void **) &dTmp1, bufSize*sizeof(float)),
		"allocating dTmp1");
	CUDAERR(cudaMalloc((void **) &dTmp2, bufSize*sizeof(float)),
		"allocating dTmp2");
	CUDAERR(cudaMalloc((void **) &dTmp3, bufSize*sizeof(float)),
		"allocating dTmp3");

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
	t = clock_start();
	processImage();
	clock_lap(t, CLOCK_OVERALL);

	// print statistics
	std::cout << "overall: " << clock_ave[CLOCK_OVERALL] << "s" << std::endl
		<< "round: " << clock_ave[CLOCK_ROUND] << "s" << std::endl
		<< "conv2d: " << clock_ave[CLOCK_CONV2D] << "s" << std::endl
		<< "mult/div: " << clock_ave[CLOCK_MULTDIV] << "s" << std::endl;

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
	CUDAERR(cudaFree(dTmp1), "freeing dTmp1");
	CUDAERR(cudaFree(dTmp2), "freeing dTmp2");
	CUDAERR(cudaFree(dTmp3), "freeing dTmp2");
	CUDAERR(cudaFree(dImgPix), "freeing dImgPix");
	free(hImgPix);
	free(t);

	// write file
	std::cout << "Writing file..." << std::endl;
	write_png_file(argv[2]);

	std::cout << "Done." << std::endl;
	return 0;
}
