#include "main.h"
#include "clock.h"

// pointwise multiplication/division of two vectors
__global__ static void pointwiseMultDiv(float *dA, float *dB, float *dC,
	int height, int rowStride, int channels, bool isMult)
{
	unsigned y, x, ind;

	// infer y, x, c from block/thread index
	y = blockDim.y * blockIdx.y + threadIdx.y;
	x = blockDim.x * blockIdx.x + threadIdx.x;
	if (y >= height || x >= rowStride) {
		return;
	}

	ind = y*rowStride + x;

	// full alpha
	if (channels==4 && x%4==3) {
		dC[ind] = 255;
		return;
	}

	dC[ind] = isMult
		? dA[ind] * dB[ind] / 255.
		: dA[ind] / max(dB[ind],1.) * 255.;	// prevent /0
}

// set image to median color
__global__ static void initImage(float *dImg, int height, int rowStride,
	int channels)
{
	unsigned y, x;

	// infer y, x, c from block/thread index
	y = blockDim.y * blockIdx.y + threadIdx.y;
	x = blockDim.x * blockIdx.x + threadIdx.x;
	if (y >= height || x >= rowStride) {
		return;
	}

	dImg[y*rowStride + x] = (channels==4 && x%4==3) ? 255 : 127;
}

// perform one "round" of LR deconvolution
__host__ static void deblurRound(float *g, unsigned fltSize)
{
	float *tmp;
	clock_t *t = clock_start();

	// convolution: tmp3 = f_i * g
	// dTmp3 = dTmp1 * flt
	conv2d<<<dimGrid, dimBlock>>>(dTmp1, g, dTmp3, channels,
		height, width, fltSize, fltSize);
	CUDAERR(cudaGetLastError(), "launch conv2d kernel 1");

	clock_lap(t, CLOCK_CONV2D);

	// pointwise division: tmp2 = c / tmp3
	// dTmp2 = dImg / dTmp3
	pointwiseMultDiv<<<dimGrid, dimBlock>>>(dImg, dTmp3, dTmp2, height,
		rowStride, channels, false);
	CUDAERR(cudaGetLastError(), "launch div kernel");

	clock_lap(t, CLOCK_MULTDIV);

	// convolution: tmp3 = tmp2 * g(-x) = tmp2 * g (g is symmetric)
	// dTmp3 = dTmp2 * g
	conv2d<<<dimGrid, dimBlock>>>(dTmp2, g, dTmp3, channels,
		height, width, fltSize, fltSize);
	CUDAERR(cudaGetLastError(), "launch conv2d kernel 2");

	clock_lap(t, CLOCK_CONV2D);

	// pointwise multiplication: tmp2 = (tmp3)(f_i)
	// dTmp2 = (dTmp3)(dTmp1)
	pointwiseMultDiv<<<dimGrid, dimBlock>>>(dTmp3, dTmp1, dTmp2, height,
		rowStride, channels, true);
	CUDAERR(cudaGetLastError(), "launch mult kernel");

	clock_lap(t, CLOCK_MULTDIV);

	// swap pointers so that f_i = dTmp1
	// dTmp2, dTmp1 = dTmp1, dTmp2
	tmp = dTmp2;
	dTmp2 = dTmp1;
	dTmp1 = tmp;

	free(t);
}

// lucy richardson deblur: deblurs what is in dImg
__host__ void deblur(int rounds, int blurSize)
{
	float *hFlt, *dFlt, *tmp;
	unsigned fltSize, i;
	clock_t *t;

	// initialize f_0 (initial estimate)
	initImage<<<dimGrid, dimBlock>>>(dTmp1, height, rowStride, channels);
	CUDAERR(cudaGetLastError(), "launch initImage kernel");

	// get initial gaussian filter
	gaussian_filter(blurSize, &hFlt, &fltSize);

	// allocate and copy filter to device
	alloc_copy_htd(hFlt, (void **) &dFlt, fltSize*fltSize*sizeof(float),
		"flt");

	// lucy-richardson iteration
	t = clock_start();
	for (i = 0; i < rounds; ++i) {
		deblurRound(dFlt, fltSize);
		clock_lap(t, CLOCK_ROUND);
	}

	// dTmp1 is currently pointing at f_i (the estimate)
	tmp = dTmp1;
	dTmp1 = dImg;
	dImg = tmp;

	// cleanup
	free(t);
	free(hFlt);
	free_d(dFlt, "dFlt");
}
