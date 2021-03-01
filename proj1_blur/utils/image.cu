#include "image.h"

// need our own allocation functions since cinfo->jpeg_memory_mgr's 
// alloced array (seems) to be tied to the lifetime of cinfo
__host__ void image_alloc_buf(struct image *img)
{
	int i, row_stride = img->width * img->components;

	img->buf_linear = (JSAMPLE *) calloc(row_stride*img->height,
		sizeof(JSAMPLE));
	img->buf = (JSAMPROW *) calloc(img->height, sizeof(JSAMPROW));

	for (i = 0; i < img->height; ++i) {
		img->buf[i] = img->buf_linear + i*row_stride;
	}
}

__host__ void image_dealloc_buf(struct image *img)
{
	// buf doesn't have to be recursively freed since it is only
	// pointing into buf_linear
	free(img->buf);
	free(img->buf_linear);
}
