#include <stdio.h>
#include "utils/jpeg.h"

__device__ void blur_kernel(int x, int y, int blur_size)
{
	// TODO
}

__host__ int main(int argc, char **argv)
{
	struct image img;
	if (argc < 3) {
		fprintf(stderr, "usage: ./%s [INPUT_FILE] [OUTPUT_FILE]\n",
			argv[0]);
		return -1;
	}

	read_jpeg_file(argv[1], &img);
	write_jpeg_file(argv[2], 100, &img);
}
