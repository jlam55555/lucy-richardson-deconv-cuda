#include "main.h"

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

	//dC[ind] = isMult ? min(dA[ind] * dB[ind] / 255., 255.) : min(dA[ind] / (dB[ind]) * 255, 255.);
	dC[ind] = isMult ? dA[ind] * dB[ind] / 255. : dA[ind] / (dB[ind]) * 255.;
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
__host__ static void deblur_round(float *g, unsigned fltSize)
{
	// TODO: fix this horrible naming convention
	float *g2, *dg2, *tmp;
	unsigned i, j;

	// calculate g2(x) = g(-x)
	/*ERR(!(g2 = (float *) malloc(fltSize*fltSize*sizeof(float))),
		"allocate g2");
	for (i = 0; i < fltSize; ++i) {
		for (j = 0; j < fltSize; ++j) {
			g2[i*fltSize+j] = g[(fltSize-1-i)*fltSize
				+(fltSize-1-j)];
		}
	}

	alloc_copy_htd(g2, (void **) &dg2, fltSize*fltSize*sizeof(float),
		"flt inverted");*/

	// g is symmetric
	g2 = g;

	// convolution: tmp3 = f_i * g
	// dTmp3 = dTmp1 * flt
	conv2d<<<dimGrid, dimBlock>>>(dTmp1, g, dTmp3, channels,
		height, width, fltSize, fltSize);
	CUDAERR(cudaGetLastError(), "launch conv2d kernel 1");

	// pointwise division: tmp2 = c / tmp3
	// dTmp2 = dImg / dTmp3
	pointwiseMultDiv<<<dimGrid, dimBlock>>>(dImg, dTmp3, dTmp2, height,
		rowStride, channels, false);
	CUDAERR(cudaGetLastError(), "launch div kernel");

	/*tmp = dTmp1;
	dTmp1 = dTmp2;
	dTmp2 = tmp;
	return;*/

	// convolution: tmp3 = tmp2 * g(-x)
	// dTmp3 = dTmp2 * g2
	conv2d<<<dimGrid, dimBlock>>>(dTmp2, g2, dTmp3, channels,
		height, width, fltSize, fltSize);
	CUDAERR(cudaGetLastError(), "launch conv2d kernel 2");

	// pointwise multiplication: tmp2 = (tmp3)(f_i)
	// dTmp2 = (dTmp3)(dTmp1)
	pointwiseMultDiv<<<dimGrid, dimBlock>>>(dTmp3, dTmp1, dTmp2, height,
		rowStride, channels, true);
	CUDAERR(cudaGetLastError(), "launch mult kernel");

	// swap pointers so that f_i = dTmp1
	// dTmp2, dTmp1 = dTmp1, dTmp2
	tmp = dTmp2;
	dTmp2 = dTmp1;
	dTmp1 = tmp;

	//free(g2);
	//free_d(dg2, "g2");
}

// lucy richardson deblur: deblurs what is in dImg
__host__ void deblur(int rounds, int blurSize)
{
	float *hFlt, *dFlt, *tmp;
	unsigned fltSize, i;

	// initialize f_0
	initImage<<<dimGrid, dimBlock>>>(dTmp1, height, rowStride, channels);
	CUDAERR(cudaGetLastError(), "launch initImage kernel");

	// get initial gaussian filter
	gaussian_filter(blurSize, &hFlt, &fltSize);

	// allocate and copy filter to device
	alloc_copy_htd(hFlt, (void **) &dFlt, fltSize*fltSize*sizeof(float),
		"flt");

	// lucy-richardson iteration
	for (i = 0; i < rounds; ++i) {
		deblur_round(dFlt, fltSize);
	}

	// dTmp1 is currently pointing at f_i (the estimate)
	tmp = dTmp1;
	dTmp1 = dImg;
	dImg = tmp;

	// cleanup
	free(hFlt);
	free_d(dFlt, "dFlt");
}
