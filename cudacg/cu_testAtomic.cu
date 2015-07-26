#include "cu_testAtomic_impl.h"
#include <bvh_common.h>

__global__ void testAtomic_kernel(int * obin,
                    int * idata,
                    int h,
                    int n)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= n) return;
	
	int d = idata[ind];
	
    atomicAdd(&obin[d/h], 1);
}

__global__ void setZero_kernel(int * obin,
                    int n)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= n) return;
	
	obin[ind] = 0;
}

__global__ void addOne_kernel(int * obin,
                    int n)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= n) return;
	
	obin[ind] += 1;
}

extern "C" {
void cu_testAtomic(int * obin,
                    int * idata,
                    int h,
                    int n)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(n, tpb);
    dim3 grid(nblk, 1, 1);
    
    testAtomic_kernel<<< grid, block>>>(obin,
        idata,
        h,
        n);
}

void cu_setZero(int * obin,
                    int n)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(n, tpb);
    dim3 grid(nblk, 1, 1);
    
    setZero_kernel<<< grid, block>>>(obin,
        n);
}

void cu_addOne(int * obin,
                    int n)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(n, tpb);
    dim3 grid(nblk, 1, 1);
    
    addOne_kernel<<< grid, block>>>(obin,
        n);
}
}
