#include "main.h"

/*
 * Review:
 * 1d convolution: (x*y)[n] = sum_i {x[i]y[n-i]}
 * 	0 <= i < w1
 * 	0 <= n-i < w2 => n-w2 < i <= n
 *		=> max(0, n-w2+1) <= i < min(w1, n+1)
 * 2d convolution: (x*y)[n,m] = sum_i {sum_j {x[i,j]y[x-i,y-j]}}
 *
 * In this case there is an additional affine transform performed so that the
 * resulting image is centered around d1's frame
 */

// performs a 2d convolution d3=d1*d2; d3 should be the same size as d1;
// assumes that d1's dimensions > d2's dimensions; this treats d2 like a
// filter, and applies it centered at each point of d1
__global__ void conv2d(float *d1, float *d2, float *d3, int ch,
	int h1, int w1, int h2, int w2)
{
	int y, x, c, i, j, imin, imax, jmin, jmax, rs, ip, jp;
	float sum;

	// infer y, x, c from block/thread index
	y = blockDim.y * blockIdx.y + threadIdx.y;
	x = blockDim.x * blockIdx.x + threadIdx.x;
	c = x % ch;
	x /= ch;

	// out of bounds, no work to do
	if (x >= w1 || y >= h1) {
		return;
	}

	// row stride (width * number of channels)
	rs = ch*w1;

	// don't mess with alpha
	if (c == 3) {
		d3[y*rs + x*ch + c] = 255;
		return;
	}

	// appropriate ranges for convolution
	imin = max(0, y+h2/2-h2+1);
	imax = min(h1, y+h2/2+1);
	jmin = max(0, x+w2/2-w2+1);
	jmax = min(w1, x+w2/2+1);

	// convolution
	// TODO: this only deals with the case where d2 has a single channel
	//	(i.e., like a filter)
	sum = 0;
	for (i = imin; i < imax; ++i) {
		for (j = jmin; j < jmax; ++j) {
			ip = i - h2/2;
			jp = j - w2/2;

			sum += d1[i*rs + j*ch + c] * d2[(y-ip)*w2 + (x-jp)];
		}
	}

	// set result
	d3[y*rs + x*ch + c] = sum;
}
