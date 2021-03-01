#include <stdio.h>
#include "jpeglib.h"

__device__ void blur_kernel(int x, int y, int blur_size)
{
	// TODO
}

__host__ void line_printer(JSAMPROW buf, int row_stride)
{
	int i;

	for (i = 0; i < row_stride; i++) {
		printf("%03d ", buf[i]);
	}
	printf("\n");
}

// using sample jpeg code from
// https://github.com/LuaDist/libjpeg/blob/master/example.c#L283
__host__ int read_jpeg_file(char *filename, void (*line_reader)(JSAMPROW, int))
{
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	FILE *infile;
	JSAMPARRAY buffer;
	int row_stride;

	if (!(infile = fopen(filename, "rb"))) {
		fprintf(stderr, "error opening %s\n", filename);
		return -1;
	}

	cinfo.err = jpeg_std_error(&jerr);

	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, infile);
	jpeg_read_header(&cinfo, TRUE);
	jpeg_start_decompress(&cinfo);
	row_stride = cinfo.output_width*cinfo.output_components;
	buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE,
		row_stride, 1);

	while (cinfo.output_scanline < cinfo.output_height) {
		jpeg_read_scanlines(&cinfo, buffer, 1);
		(*line_reader)(buffer[0], row_stride);
	}

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(infile);

	return 0;
}

__host__ int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "no input file specified\n");
		return -1;
	}

	read_jpeg_file(argv[1], line_printer);
}
