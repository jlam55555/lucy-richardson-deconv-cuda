#include "main.h"

// allocate a memory region on the device and copy a value from the host
__host__ void alloc_copy_htd(void *hptr, void **dptr, unsigned size,
	std::string name)
{
	CUDAERR(cudaMalloc(dptr, size), "allocating " << name <<);
	CUDAERR(cudaMemcpy(*dptr, hptr, size, cudaMemcpyHostToDevice),
		"copying " << name << " to device");
}

// copy memory region from device to host (if hptr specified)
__host__ void copy_dth(void *hptr, void *dptr, unsigned size,
	std::string name)
{
	CUDAERR(cudaMemcpy(hptr, dptr, size, cudaMemcpyDeviceToHost),
		"copying " << name <<);
}

// free emory region on device
__host__ void free_d(void *dptr, std::string name)
{
	CUDAERR(cudaFree(dptr), "freeing " << name <<);
}
