#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pngio.h"
#include "main.h"

#define CLIP(x, min, max) (x)<(min)?(min):(x)>(max)?(max):(x);

float *dImg, *dTmp1, *dTmp2, *dTmp3;
unsigned int rowStride, channels, bufSize, blockSize;

// create gaussian kernel: see comments from cuda implementation
float *gaussian_filter(float blurStd, unsigned *fltSizep)
{
	float fltSum, cent, *flt;
	unsigned i, j, dim;

	dim = 6*blurStd+1;
	cent = (dim-1.)/2;

	flt = malloc(dim*dim*sizeof(float));
	fltSum = 0;
	for (i = 0; i < dim; ++i) {
		for (j = 0; j < dim; ++j) {
			flt[i*dim+j] = exp(-(pow(i-cent,2)+pow(j-cent,2))
				/(2*blurStd*blurStd))/(2*M_PI*blurStd*blurStd);
			fltSum += flt[i*dim+j];
		}
	}

	// normalize the filter
	for (i = 0; i < dim*dim; ++i) {
		flt[i] /= fltSum;
	}

	*fltSizep = dim;
	return flt;
}

// see comment on the cuda version of this function
// assume only three channels, no alpha channel
void conv2d(float *d1, float *d2, float *d3,
	int ch, int h1, int w1, int h2, int w2)
{
	int y, x, rs, c, imin, imax, jmin, jmax, i, j, ip, jp, h22, w22;
	float sum;

	h22 = h2 >> 1;	// h2 / 2
	w22 = w2 >> 1;	// w2 / 2
	rs = w1 * 3;	// row stride

	for (y = 0; y < h1; ++y) {
	for (x = 0; x < w1; ++x) {
	for (c = 0; c < 3; ++c) {

		// get appropriate ranges for convolution
		// don't use min/max macros to avoid recomputation
		imin = y+h22-h2+1;
		if (imin < 0) imin = 0;
		imax = y+h22+1;
		if (imax > h1) imax = h1;
		jmin = x+w22-w2+1;
		if (jmin < 0) jmin = 0;
		jmax = x+w22+1;
		if (jmax > w1) jmax = w1;

		sum = 0;
		for (i = imin; i < imax; ++i) {
		for (j = jmin; j < jmax; ++j) {
			// transformed i, j for h2
			ip = i - h22;
			jp = j - w22;

			sum += d1[i*rs + j*ch + c] * d2[(y-ip)*w2 + (x-jp)];
		}
		}

		d3[y*rs + x*ch + c] = sum;
	}
	}
	}
}

// multiply or divide two pixels elementwise
void pointwiseMultDiv(float *dA, float *dB, float *dC, int size, int isMult)
{
	unsigned i;

	// put conditional on outside to reduce computational cost
	if (isMult) {
		for (i = 0; i < size; ++i) {
			dC[i] = dA[i] * dB[i] / 255.;
		}
	} else {
		for (i = 0; i < size; ++i) {
			// prevent divide by zero
			dC[i] = dA[i] / (dB[i] < 1 ? 1 : dB[i]) * 255.;
		}
	}
}

// see explanation in cuda implementation
void deblurRound(float *g, unsigned fltSize)
{
	float *tmp;

	conv2d(dTmp1, g, dTmp3, 3, height, width, fltSize, fltSize);

	pointwiseMultDiv(dImg, dTmp3, dTmp2, bufSize, 0);

//	tmp = dTmp2;
//	dTmp2 = dTmp1;
//	dTmp1 = tmp;
//	return;

	conv2d(dTmp2, g, dTmp3, 3, height, width, fltSize, fltSize);

	pointwiseMultDiv(dTmp3, dTmp1, dTmp2, bufSize, 1);

	// swap pointers
	tmp = dTmp2;
	dTmp2 = dTmp1;
	dTmp1 = tmp;
}

// lucy-richardson deconvolution with a gaussian kernel
void deblur(int iterations, int blurStd)
{
	float *flt, *tmp;
	int fltSize, i;

	// create gaussian filter
	flt = gaussian_filter(blurStd, &fltSize);

	// initialize estimate
	for (i = 0; i < bufSize; ++i) {
		dTmp1[i] = 127;
	}

	// iterate!
	for (i = 0; i < iterations; ++i) {
		printf("Round %d\n", i);
		deblurRound(flt, fltSize);
	}

	// img currently in dTmp1, move to dImg
	tmp = dTmp1;
	dTmp1 = dImg;
	dImg = tmp;

	// cleanup
	free(flt);
}

void processImage()
{
	deblur(25, 2);
}

int main(int argc, char **argv)
{
	unsigned y, x;

	// get input file from stdin
	if (argc < 2) {
		printf("missing input file as cmd parameter\n"
			"\tusage: ./deblur [INPUT_FILE].png\n");
		return -1;
	}

	// read input file
	printf("Reading file...\n");
	read_png_file(argv[1]);

	// assume only RGB (3 channels) or RGBA (4 channels)
	channels = color_type==PNG_COLOR_TYPE_RGBA ? 4 : 3;
	rowStride = width * channels;
	bufSize = (width * 3) * height;

	dImg = malloc(bufSize * sizeof(float));
	dTmp1 = malloc(bufSize * sizeof(float));
	dTmp2 = malloc(bufSize * sizeof(float));
	dTmp3 = malloc(bufSize * sizeof(float));

	// copy image to contiguous buffer (double pointer is not guaranteed
	// to be contiguous) and convert to float; also only copy over rgb
	// (not alpha)
	for (y = 0; y < height; ++y) {
		for (x = 0; x < width; ++x) {
			dImg[y*width*3 + x*3]
				= (float) row_pointers[y][x*channels];
			dImg[y*width*3 + x*3+1]
				= (float) row_pointers[y][x*channels+1];
			dImg[y*width*3 + x*3+2]
				= (float) row_pointers[y][x*channels+2];
		}
	}

	// begin processing
	printf("Processing image...\n");
	processImage();

	// copy image back from dImg; leave alpha channel as is
	for (y = 0; y < height; ++y) {
		for (x = 0; x < width; ++x) {
			row_pointers[y][x*channels]
				= CLIP(dImg[y*width*3 + x*3], 0, 255);
			row_pointers[y][x*channels+1]
				= CLIP(dImg[y*width*3 + x*3+1], 0, 255);
			row_pointers[y][x*channels+2]
				= CLIP(dImg[y*width*3 + x*3+2], 0, 255);
		}
	}

	// write file
	printf("Writing image...\n");
	write_png_file("out.png");

	// cleanup
	free(dImg);
	free(dTmp1);
	free(dTmp2);
	free(dTmp3);

	printf("Done\n");
	return 0;
}
