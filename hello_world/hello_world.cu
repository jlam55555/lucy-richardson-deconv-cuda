#include <stdio.h>

__global__ void vec_add_kernel(float *a, float *b, float *c, int n)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

__host__ void vec_add(float *h_a, float *h_b, float *h_c, int n)
{
	int size = n*sizeof(float);
	float *d_a, *d_b, *d_c;

	// allocate memory and initialize on device
	cudaMalloc((void **) &d_a, size);
	cudaMalloc((void **) &d_b, size);
	cudaMalloc((void **) &d_c, size);

	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	// calculate
	vec_add_kernel<<<ceil(n/256.0), 256>>>(d_a, d_b, d_c, n);

	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

	// cleanup
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

__host__ int main(void)
{
	// create data to add
	float h_a[] = {1,2,3};
	float h_b[] = {1,2,3};
	float h_c[3];
	int n = 3;

	vec_add(h_a, h_b, h_c, n);

	for (int i = 0; i < 3; ++i) {
		printf("%f\n", h_c[i]);
	}
}
