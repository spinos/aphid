#include <cuda_runtime_api.h>

__global__ void quickSort_kernel(int * obin,
                    int * idata,
                    int h,
                    int n)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= n) return;
	
	int d = idata[ind];
	
    atomicAdd(&obin[d/h], 1);
}

