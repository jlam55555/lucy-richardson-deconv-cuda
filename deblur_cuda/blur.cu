#include "main.h"

// returns a circular gaussian filter with standard deviation blurStd; returns
// the dimensions of the filter in fltSize
__host__ void gaussian_filter(float blurStd, float **fltp, unsigned *fltSizep)
{
	float fltSum, cent, *flt;
	unsigned i, j, fltSize;

	// initialize filter; 3x3 circular gaussian filter
	// https://en.wikipedia.org/wiki/Gaussian_blur
	fltSize = 6*blurStd+1;	// for factor of 6 see Wikipedia
				// +1 to make it odd for better centering
	cent = (fltSize-1.)/2;	// center of filter

	ERR(!(flt = (float *) malloc(fltSize*fltSize*sizeof(float))),
		"allocate flt");
	fltSum = 0;
	for (i = 0; i < fltSize; ++i) {
		for (j = 0; j < fltSize; ++j) {
			flt[i*fltSize+j] = exp(-(pow(i-cent,2)+pow(j-cent,2))
				/(2*blurStd*blurStd))/(2*M_PI*blurStd*blurStd);
			fltSum += flt[i*fltSize+j];
		}
	}

	// normalize the filter
	for (i = 0; i < fltSize*fltSize; ++i) {
		flt[i] /= fltSum;
	}

	*fltSizep = fltSize;
	*fltp = flt;
}

// performs a gaussian blur on an image
__host__ void blur(int blurSize)
{
	float *hFlt, *dFlt, *tmp;
	unsigned fltSize;

	gaussian_filter(blurSize, &hFlt, &fltSize);

	// allocate and copy filter to device
	alloc_copy_htd(hFlt, (void **) &dFlt, fltSize*fltSize*sizeof(float),
		"flt");

	// blur image (for testing)
	conv2d<<<dimGrid, dimBlock>>>(dImg, dFlt, dTmp1, channels,
		height, width, fltSize, fltSize);

	// result is currently in dTmp1, swap pointers
	tmp = dImg;
	dImg = dTmp1;
	dTmp1 = tmp;

	// cleanup
	free(hFlt);
	free_d(dFlt, "dFlt");
}
