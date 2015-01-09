#include "createBvh_implement.h"

inline __device__ void resetAabb(Aabb & dst)
{
    dst.low = make_float3(10e8, 10e8, 10e8);
    dst.high = make_float3(-10e8, -10e8, -10e8);
}

inline __device__ void expandAabb(Aabb & dst, float4 p)
{
    if(p.x < dst.low.x) dst.low.x = p.x;
    if(p.y < dst.low.y) dst.low.y = p.y;
    if(p.z < dst.low.z) dst.low.z = p.z;
    if(p.x > dst.high.x) dst.high.x = p.x;
    if(p.y > dst.high.y) dst.high.y = p.y;
    if(p.z > dst.high.z) dst.high.z = p.z;
}

__global__ void calculateAabbs_kernel(Aabb *dst, float4 * cvs, EdgeContact * edges, unsigned maxEdgeInd, unsigned maxVertInd)
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= maxEdgeInd) return;
	
	EdgeContact e = edges[idx];
	unsigned v0 = e.v[0];
	unsigned v1 = e.v[1];
	unsigned v2 = e.v[2];
	unsigned v3 = e.v[3];
	
	Aabb res;
	resetAabb(res);
	if(v0 < maxVertInd) expandAabb(res, cvs[v0]);
	if(v1 < maxVertInd) expandAabb(res, cvs[v1]);
	if(v2 < maxVertInd) expandAabb(res, cvs[v2]);
	if(v3 < maxVertInd) expandAabb(res, cvs[v3]);
	
	dst[idx] = res;
}

extern "C" void bvhCalculateAabbs(Aabb *dst, float4 * cvs, EdgeContact * edges, unsigned numEdges, unsigned numVertices)
{
    dim3 block(256, 1, 1);
    unsigned nblk = iDivUp(numEdges, 256);
    
    dim3 grid(nblk, 1, 1);
    calculateAabbs_kernel<<< grid, block >>>(dst, cvs, edges, numEdges, numVertices);
}
