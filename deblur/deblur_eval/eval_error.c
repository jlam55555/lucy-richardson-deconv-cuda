#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "pngio.h"

typedef unsigned char byte;

// calculates mean squared error per pixel channel
float calculateError(byte *img1, byte *img2, unsigned size)
{
	unsigned i, err;

	err = 0;
	for (i = 0; i < size; ++i) {
		err += pow(img1[i] - img2[i], 2);
	}
	return ((float) err) / size;
}

int main(int argc, char **argv)
{
	byte *img1, *img2;
	unsigned channels, bufSize, j, i, k;

	// get files
	if (argc < 3) {
		printf("usage: %s ORIGINAL.png IMG1.png [IMG2.png ...]\n"
			"\tcompares ORIGINAL.png to each of IMG1.png,"
			" IMG2.png, ...\n", argv[0]);
		return -1;
	}

	// read in original and copy to buffer
	read_png_file(argv[1]);

	channels = color_type==PNG_COLOR_TYPE_RGBA ? 4 : 3;
	bufSize = (width * 3) * height;

	img1 = malloc(bufSize);
	img2 = malloc(bufSize);

	// only copy rgb values
	for (j = 0; j < height; ++j) {
	for (i = 0; i < width; ++i) {
		img1[j*width*3 + i*3] = row_pointers[j][i*channels];
		img1[j*width*3 + i*3+1] = row_pointers[j][i*channels+1];
		img1[j*width*3 + i*3+2] = row_pointers[j][i*channels+2];
	}
	}

	for (k = 2; k < argc; ++k) {
		read_png_file(argv[k]);

		channels = color_type==PNG_COLOR_TYPE_RGBA ? 4 : 3;

		for (j = 0; j < height; ++j) {
		for (i = 0; i < width; ++i) {
			img2[j*width*3 + i*3] = row_pointers[j][i*channels];
			img2[j*width*3 + i*3+1] = row_pointers[j][i*channels+1];
			img2[j*width*3 + i*3+2] = row_pointers[j][i*channels+2];
		}
		}

		// calculate and print error
		printf("MSE %s <-> %s (per datapoint): %f\n",
			argv[1], argv[k],
			calculateError(img1, img2, bufSize));
	}

	// cleanup
	free(img1);
	free(img2);

	return 0;
}
