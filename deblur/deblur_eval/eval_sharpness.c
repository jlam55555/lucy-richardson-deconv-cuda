#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "pngio.h"

typedef unsigned char byte;

// see comment on the cuda version of this function
// assume only three channels, no alpha channel
// this is simplified a little to take byte inputs (but a float output)
void conv2d(byte *d1, float *d2, float *d3,
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

// calculate measure of sharpness/blur using an edge-detection laplacian filter;
// laplacian turns edges into elements with larger magnitude, so we simply
// calculate the mean of the values after passing the edge filter
// through (all data channels)
//
// see: https://stackoverflow.com/a/7767755
float measureSharpness(byte *img, int channels, int height, int width)
{
	unsigned size = channels * height * width, i;
	float contrast, *convOut;

	// laplacian filter from https://stackoverflow.com/a/7766036
	float flt[] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
	unsigned fltSize = 3;

	// convolve image with laplacian filter
	convOut = malloc(size * sizeof(float));
	conv2d(img, flt, convOut, channels, height, width, fltSize, fltSize);

	// sum over values in convolution output
	contrast = 0;
	for (i = 0; i < size; ++i) {
		contrast += pow(convOut[i], 2);
	}

	// cleanup
	free(convOut);

	return contrast / size;
}

int main(int argc, char **argv)
{
	byte *img;
	unsigned channels, bufSize, j, i, k;

	// get files
	if (argc < 2) {
		printf("usage: %s IMG.png [IMG2.png ...]\n"
			"\tmeasures the sharpness of each image\n", argv[0]);
		return -1;
	}

	for (k = 1; k < argc; ++k) {
		// read in image and copy to buffer
		read_png_file(argv[k]);

		channels = color_type==PNG_COLOR_TYPE_RGBA ? 4 : 3;
		bufSize = (width * 3) * height;

		img = malloc(bufSize);

		// only copy rgb values
		for (j = 0; j < height; ++j) {
		for (i = 0; i < width; ++i) {
			img[j*width*3 + i*3] = row_pointers[j][i*channels];
			img[j*width*3 + i*3+1] = row_pointers[j][i*channels+1];
			img[j*width*3 + i*3+2] = row_pointers[j][i*channels+2];
		}
		}

		// calculate and print sharpness estimate
		printf("Sharpness measure of %s: %f\n", argv[k],
			measureSharpness(img, 3, height, width));

		// cleanup
		free(img);
	}

	return 0;
}
