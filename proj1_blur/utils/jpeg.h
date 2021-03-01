#ifndef JPEGH
#define JPEGH

#include "image.h"

__host__ int write_jpeg_file(char *filename, int quality, struct image *img);
__host__ int read_jpeg_file(char *filename, struct image *img);

#endif
