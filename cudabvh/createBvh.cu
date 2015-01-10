#include "createBvh_implement.h"

// Expands a 10-bit integer into 30 bits 
// by inserting 2 zeros after each bit. 
inline __device__ uint expandBits(uint v) 
{ 
    v = (v * 0x00010001u) & 0xFF0000FFu; 
    v = (v * 0x00000101u) & 0x0F00F00Fu; 
    v = (v * 0x00000011u) & 0xC30C30C3u; 
    v = (v * 0x00000005u) & 0x49249249u; 
    return v; 
}

// Calculates a 30-bit Morton code for the 
// given 3D point located within the unit cube [0,1].
inline __device__ uint morton3D(float x, float y, float z) 
{ 
    x = min(max(x * 1024.0f, 0.0f), 1023.0f); 
    y = min(max(y * 1024.0f, 0.0f), 1023.0f); 
    z = min(max(z * 1024.0f, 0.0f), 1023.0f); 
    uint xx = expandBits((uint)x); 
    uint yy = expandBits((uint)y); 
    uint zz = expandBits((uint)z); 
    return xx * 4 + yy * 2 + zz; 
} 

inline __device__ void normalizeByBoundary(float & x, float low, float high)
{
	if(x < low) x = 0.0;
	else if(x >= high) x = 1.0;
	else {
		float dx = high - low;
		if(dx < TINY_VALUE) x = 0.0;
		else x = (x - low) / dx;
	}
}

inline __device__ void resetAabb(Aabb & dst)
{
    dst.low = make_float3(10e9, 10e9, 10e9);
    dst.high = make_float3(-10e9, -10e9, -10e9);
}

inline __device__ void expandAabb(Aabb & dst, float3 p)
{
    if(p.x < dst.low.x) dst.low.x = p.x - TINY_VALUE;
    if(p.y < dst.low.y) dst.low.y = p.y - TINY_VALUE;
    if(p.z < dst.low.z) dst.low.z = p.z - TINY_VALUE;
    if(p.x > dst.high.x) dst.high.x = p.x + TINY_VALUE;
    if(p.y > dst.high.y) dst.high.y = p.y + TINY_VALUE;
    if(p.z > dst.high.z) dst.high.z = p.z + TINY_VALUE;
}

inline __device__ float3 centroidOfAabb(const Aabb & box)
{
	return make_float3(box.low.x * 0.5 + box.high.x * 0.5, box.low.y * 0.5 + box.high.y * 0.5, box.low.z * 0.5 + box.high.z * 0.5);
}

__global__ void calculateAabbs_kernel(Aabb *dst, float3 * cvs, EdgeContact * edges, unsigned maxEdgeInd, unsigned maxVertInd)
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

__global__ void calculateLeafHash_kernel(KeyValuePair *dst, Aabb * leafBoxes, uint maxInd, Aabb boundary)
{
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= maxInd) return;
	
	float3 c = centroidOfAabb(leafBoxes[idx]);
	normalizeByBoundary(c.x, boundary.low.x, boundary.high.x);
	normalizeByBoundary(c.y, boundary.low.y, boundary.high.y);
	normalizeByBoundary(c.z, boundary.low.z, boundary.high.z);
	
	dst[idx].key = morton3D(c.x, c.y, c.z);
	dst[idx].value = idx;
}

extern "C" void bvhCalculateLeafAabbs(Aabb *dst, float3 * cvs, EdgeContact * edges, unsigned numEdges, unsigned numVertices)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numEdges, 512);
    
    dim3 grid(nblk, 1, 1);
    calculateAabbs_kernel<<< grid, block >>>(dst, cvs, edges, numEdges, numVertices);
}

extern "C" void bvhCalculateLeafHash(KeyValuePair * dst, Aabb * leafBoxes, uint numLeaves, Aabb bigBox)
{
	dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numLeaves, 512);
    
    dim3 grid(nblk, 1, 1);
	calculateLeafHash_kernel<<< grid, block >>>(dst, leafBoxes, numLeaves, bigBox);
}
