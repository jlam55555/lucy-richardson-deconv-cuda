#include <stdio.h>
#include "utils/jpeg.h"

__global__ void blur_kernel(JSAMPLE *in_buf, JSAMPLE *out_buf,
	int width, int height, int components, int blur_radius)
{
	int col_pos = blockIdx.x * blockDim.x + threadIdx.x,
	    row = blockIdx.y * blockDim.y + threadIdx.y,
	    col = col_pos/components, component = col_pos%components,
	    x, y, x_max, y_max, pix_value, pix_count;

	for (x=max(0,col-blur_radius), x_max=min(width-1,col+blur_radius),
		pix_value=pix_count=0; x < x_max; ++x) {

		for (y=max(0,row-blur_radius),
			y_max=min(height-1,row+blur_radius); y < y_max; ++y) {

			pix_value += in_buf[(y*width+x)*components+component]; 
			++pix_count;
		}
	}

	out_buf[row*width*components+col_pos] = pix_value/pix_count;
}

__host__ void blur(struct image *img, int blur_radius)
{
	JSAMPLE *d_in_buf, *d_out_buf;
	int row_stride = img->width * img->components,
	    size = row_stride * img->height * sizeof(JSAMPLE),
	    bs = 10;

	// alloc memory for image and transformed image
	cudaMalloc((void **) &d_in_buf, size);
	cudaMalloc((void **) &d_out_buf, size);

	// copy image
	cudaMemcpy(d_in_buf, img->buf_linear, size, cudaMemcpyHostToDevice);

	// blur
	dim3 dimGrid(ceil(row_stride*1./bs), ceil(img->height*1./bs), 1);
	dim3 dimBlock(bs, bs, 1);
	blur_kernel<<<dimGrid, dimBlock>>>(d_in_buf, d_out_buf,
		img->width, img->height, img->components, blur_radius);

	// return image
	cudaMemcpy(img->buf_linear, d_out_buf, size, cudaMemcpyDeviceToHost);

	// cleanup
	cudaFree(d_in_buf);
	cudaFree(d_out_buf);
}

__host__ int main(int argc, char **argv)
{
	struct image img[1];
	if (argc < 4) {
		fprintf(stderr, "usage: ./%s [INPUT_FILE] [OUTPUT_FILE]"
			" [BLUR_RADIUS]\n", argv[0]);
		return -1;
	}

	fprintf(stdout, "Reading file...\n");
	read_jpeg_file(argv[1], img);

	fprintf(stdout, "Blurring...\n");
	blur(img, atoi(argv[3]));

	fprintf(stdout, "Writing file...\n");
	write_jpeg_file(argv[2], 100, img);
	return 0;
}
