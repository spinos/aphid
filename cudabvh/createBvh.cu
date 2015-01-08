#include "createBvh_implement.h"

inline __device__ void resetAabb(Aabb & dst)
{
    dst.low = make_float3(10e8, 10e8, 10e8);
    dst.high = make_float3(-10e8, -10e8, -10e8);
}

inline __device__ void expandAabb(Aabb & dst, float3 p)
{
    if(p.x < dst.low.x) dst.low.x = p.x;
    if(p.y < dst.low.y) dst.low.y = p.y;
    if(p.z < dst.low.z) dst.low.z = p.z;
    if(p.x > dst.high.x) dst.high.x = p.x;
    if(p.y > dst.high.y) dst.high.y = p.y;
    if(p.z > dst.high.z) dst.high.z = p.z;
}

__global__ void calculateAabbs_kernel(Aabb *dst, float3 * cvs, unsigned * indices, unsigned width, unsigned maxIdx)
{
    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;

	unsigned idx = y*width+x;
	if(idx >= maxIdx) return;
	
	unsigned v0 = indices[idx * 3];
	unsigned v1 = indices[idx * 3 + 1];
	unsigned v2 = indices[idx * 3 + 2];
	
	Aabb res;
	resetAabb(res);
	expandAabb(res, cvs[v0]);
	expandAabb(res, cvs[v1]);
	expandAabb(res, cvs[v2]);
	
	dst[idx] = res;
}

extern "C" void calculateAabbs(Aabb *dst, float3 * cvs, unsigned * indices, unsigned numTriangle)
{
    dim3 block(8, 8, 1);
    unsigned nblk = iDivUp(numTriangle, 64);
    unsigned width = nblk * 8;
    
    dim3 grid(nblk, 1, 1);
    calculateAabbs_kernel<<< grid, block >>>(dst, cvs, indices, width, numTriangle);
}
