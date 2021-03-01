#include "jpeg.h"

// using sample jpeg code from
// https://github.com/LuaDist/libjpeg/blob/master/example.c#L283
__host__ int write_jpeg_file(char *filename, int quality, struct image *img)
{
	struct jpeg_compress_struct cinfo;	
	struct jpeg_error_mgr jerr;
	FILE *outfile;

	if (!(outfile = fopen(filename, "wb"))) {
		fprintf(stderr, "can't open %s for writing\n", filename);
		return -1;
	}

	// init compression process
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, outfile);

	// set compression parameters
	cinfo.image_width = img->width;
	cinfo.image_height = img->height;
	cinfo.input_components = img->components;
	cinfo.in_color_space = JCS_RGB;

	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, quality, TRUE);
	jpeg_start_compress(&cinfo, TRUE);

	// write image
	jpeg_write_scanlines(&cinfo, img->buf, img->height);

	// cleanup
	jpeg_finish_compress(&cinfo);
	fclose(outfile);
	jpeg_destroy_compress(&cinfo);
	return 0;
}

__host__ int read_jpeg_file(char *filename, struct image *img)
{
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	FILE *infile;

	if (!(infile = fopen(filename, "rb"))) {
		fprintf(stderr, "error opening %s\n", filename);
		return -1;
	}

	// init decompress process
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, infile);
	jpeg_read_header(&cinfo, TRUE);
	jpeg_start_decompress(&cinfo);

	// allocate buffer
	img->width = cinfo.image_width;
	img->height = cinfo.image_height;
	img->components = cinfo.output_components;
	image_alloc_buf(img);

	while (cinfo.output_scanline < img->height) {
		jpeg_read_scanlines(&cinfo, img->buf+cinfo.output_scanline,
			img->height-cinfo.output_scanline);
	}

	// cleanup
	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(infile);

	return 0;
}
