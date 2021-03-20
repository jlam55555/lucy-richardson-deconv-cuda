#include <stdio.h>
#include "pngio.h"
#include "main.h"

#define CLIP(x, min, max) (x)<(min)?(min):(x)>(max)?(max):(x);

float *dImg, *dTmp1, *dTmp2, *dTmp3;
unsigned int rowStride, channels, bufSize, blockSize;

void processImage()
{
	// TODO
}

int main(int argc, char **argv)
{
	unsigned y, x;

	// get input file from stdin
	if (argc < 2) {
		printf("missing input file as cmd parameter\n"
			"\tusage: ./deblur [INPUT_FILE].png");
		return -1;
	}

	// read input file
	printf("Reading file...\n");
	read_png_file(argv[1]);

	// assume only RGB (3 channels) or RGBA (4 channels)
	channels = color_type==PNG_COLOR_TYPE_RGBA ? 4 : 3;
	rowStride = width * channels;
	bufSize = rowStride * height;

	dImg = malloc(bufSize * sizeof(float));
	dTmp1 = malloc(bufSize * sizeof(float));
	dTmp2 = malloc(bufSize * sizeof(float));
	dTmp3 = malloc(bufSize * sizeof(float));

	// copy image to contiguous buffer (double pointer is not guaranteed
	// to be contiguous) and convert to float; also only copy over rgb
	// (not alpha)
	for (y = 0; y < height; ++y) {
		for (x = 0; x < width; ++x) {
			dImg[y*rowStride + x*channels]
				= row_pointers[y][x*channels];
			dImg[y*rowStride + x*channels+1]
				= row_pointers[y][x*channels+2];
			dImg[y*rowStride + x*channels+2]
				= row_pointers[y][x*channels+2];
		}
	}

	// begin processing
	printf("Processing image...\n");
	processImage();

	// copy image back from dImg
	for (y = 0; y < height; ++y) {
		for (x = 0; x < width; ++x) {
			row_pointers[y][x*channels]
				= CLIP(dImg[y*rowStride + x*channels], 0, 255);
			row_pointers[y][x*channels+2]
				= CLIP(dImg[y*rowStride + x*channels+1], 0, 255);
			row_pointers[y][x*channels+2]
				= CLIP(dImg[y*rowStride + x*channels+2], 0, 255);
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
