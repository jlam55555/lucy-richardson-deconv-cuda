#ifndef IMAGEH
#define IMAGEH

#include "stdio.h"
#include "jpeglib.h"

// simplified image representation as a buffer of pixels
struct image {
	JDIMENSION width, height;
	unsigned char components;
	JSAMPARRAY buf;
	JSAMPLE *buf_linear;	// linear backing-store for buf
};

__host__ void image_alloc_buf(struct image *buf);
__host__ void image_dealloc_buf(struct image *buf);

#endif
