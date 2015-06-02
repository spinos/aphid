#ifndef BVHHASH_CUH
#define BVHHASH_CUH

#include <cuda_runtime_api.h>
#include "bvh_math.cuh"
#include "radixsort_implement.h"
#include "Aabb.cuh"

inline __device__ uint expandBits(uint v) 
{ 
    v = (v * 0x00010001u) & 0xFF0000FFu; 
    v = (v * 0x00000101u) & 0x0F00F00Fu; 
    v = (v * 0x00000011u) & 0xC30C30C3u; 
    v = (v * 0x00000005u) & 0x49249249u; 
    return v; 
}

inline __device__ uint morton3D(float x, float y, float z) 
{ 
    x = min(max(x * 1024.0f, 0.0f), 1023.0f); 
    y = min(max(y * 1024.0f, 0.0f), 1023.0f); 
    z = min(max(z * 1024.0f, 0.0f), 1023.0f); 
    uint xx = expandBits((uint)x); 
    uint yy = expandBits((uint)y); 
    uint zz = expandBits((uint)z); 
    return (xx <<2) + (yy <<1) + zz; 
} 

inline __device__ void normalizeByBoundary(float & x, float low, float width)
{
	if(x < low) x = 0.0f;
	else if(x > low + width) x = 1.0f;
	else {
		if(width < TINY_VALUE2) x = 0.0f;
		else x = (x - low) / width;
	}
}

__global__ void calculateLeafHash_kernel(KeyValuePair *dst, Aabb * leafBoxes, uint maxInd, uint bufSize, 
		Aabb * boundary)
{
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(idx >= bufSize) return;
	
	dst[idx].value = idx;
	if(idx >= maxInd) {
		dst[idx].key = 1073741823; // 2^30 - 1
		return;
	}
	
	float3 c = centroidOfAabb(leafBoxes[idx]);
	
	float side = longestSideOfAabb(*boundary);
	float3 d = float3_difference(boundary->high, boundary->low);
	if(d.x < side) d.x = side;
	if(d.y < side) d.y = side;
	if(d.z < side) d.z = side;
	normalizeByBoundary(c.x, boundary->low.x, d.x);
	normalizeByBoundary(c.y, boundary->low.y, d.y);
	normalizeByBoundary(c.z, boundary->low.z, d.z);
	
	dst[idx].key = morton3D(c.x, c.y, c.z);
}
#endif        //  #ifndef BVHHASH_CUH

