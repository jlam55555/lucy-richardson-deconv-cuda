#include <cmath>
#include "main.h"
#include "pngio.h"

// these are declared in main.h
cudaError_t err = cudaSuccess;
float *dImg, *dTmp;
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

// image processing routines go here
__host__ static void processImage(void)
{
	float *hFlt, *dFlt, *tmp;
	unsigned fltSize;

/*
	// invert image (for testing)
	invert<<<dimGrid, dimBlock>>>(dImg, height, rowStride,
		color_type==PNG_COLOR_TYPE_RGBA);
	CUDAERR(cudaGetLastError(), "launch invert kernel");
*/
	gaussian_filter(3, &hFlt, &fltSize);

	// allocate and copy filter to device
	alloc_copy_htd(hFlt, (void **) &dFlt, fltSize*fltSize*sizeof(float),
		"flt");

	// blur image (for testing)
	conv2d<<<dimGrid, dimBlock>>>(dImg, dFlt, dTmp, channels,
		height, width, fltSize, fltSize);

	// result is currently in dTmp, swap pointers
	tmp = dImg;
	dImg = dTmp;
	dTmp = tmp;

	// cleanup
	free(hFlt);
	free_d(dFlt, "dFlt");
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
	processImage();

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
