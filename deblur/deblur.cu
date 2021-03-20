#include "main.h"

// set image to median color
__global__ static void init_image(float *dImg, int height, int rowStride,
	int ch)
{
	unsigned y, x;

	// infer y, x, c from block/thread index
	y = blockDim.y * blockIdx.y + threadIdx.y;
	x = blockDim.x * blockIdx.x + threadIdx.x;
	if (y >= height || x >= rowStride || (ch==4 && x%4==3)) {
		return;
	}
	dImg[y*rowStride + x] = 127;
}

// perform one "round" of LR deconvolution
__host__ static void deblur_round(float *flt)
{

}

// lucy richardson deblur: deblurs what is in dImg
__host__ void deblur(void)
{
	float *hFlt, *dFlt;
	unsigned blurSize = 5, fltSize;

	// initialize f_0
	init_image<<<dimGrid, dimBlock>>>(dTmp1, height, rowStride, channels);

	// get initial gaussian filter
	gaussian_filter(blurSize, &hFlt, &fltSize);

	// allocate and copy filter to device
	alloc_copy_htd(hFlt, (void **) &dFlt, fltSize*fltSize*sizeof(float),
		"flt");

	deblur_round(nullptr);

	// cleanup
	free(hFlt);
	free_d(dFlt, "dFlt");
}
