#ifndef AABB_CUH
#define AABB_CUH

#include <bvh_common.h>

inline __device__ void resetAabb(Aabb & dst)
{
    dst.low = make_float3(HUGE_VALUE, HUGE_VALUE, HUGE_VALUE);
    dst.high = make_float3(-HUGE_VALUE, -HUGE_VALUE, -HUGE_VALUE);
}

inline __device__ void expandAabb(Aabb & dst, float3 b)
{
    dst.low.x = min(dst.low.x, b.x);
    dst.low.y = min(dst.low.y, b.y);
    dst.low.z = min(dst.low.z, b.z);
    dst.high.x = max(dst.high.x, b.x);
    dst.high.y = max(dst.high.y, b.y);
    dst.high.z = max(dst.high.z, b.z);
}

inline __device__ void expandAabb(Aabb & dst, const Aabb & b)
{
	dst.low.x = min(dst.low.x, b.low.x);
    dst.low.y = min(dst.low.y, b.low.y);
    dst.low.z = min(dst.low.z, b.low.z);
    dst.high.x = max(dst.high.x, b.high.x);
    dst.high.y = max(dst.high.y, b.high.y);
    dst.high.z = max(dst.high.z, b.high.z);
}

inline __device__ void resetAabb(Aabb4 & dst)
{
    dst.low.x = HUGE_VALUE;
    dst.low.y = HUGE_VALUE;
    dst.low.x = HUGE_VALUE;
    dst.high.y = -HUGE_VALUE;
    dst.high.x = -HUGE_VALUE;
    dst.high.z = -HUGE_VALUE;
}

inline __device__ void expandAabb(Aabb4 & dst, const Aabb4 & b)
{
    dst.low.x = min(dst.low.x, b.low.x);
    dst.low.y = min(dst.low.y, b.low.y);
    dst.low.z = min(dst.low.z, b.low.z);
    dst.high.x = max(dst.high.x, b.high.x);
    dst.high.y = max(dst.high.y, b.high.y);
    dst.high.z = max(dst.high.z, b.high.z);
}

inline __device__ void expandAabb(Aabb4 & dst, const float4 & b)
{
    dst.low.x = min(dst.low.x, b.x);
    dst.low.y = min(dst.low.y, b.y);
    dst.low.z = min(dst.low.z, b.z);
    dst.high.x = max(dst.high.x, b.x);
    dst.high.y = max(dst.high.y, b.y);
    dst.high.z = max(dst.high.z, b.z);
}

inline __device__ float3 centroidOfAabb(const Aabb & box)
{
	return make_float3(box.low.x * 0.5 + box.high.x * 0.5, box.low.y * 0.5 + box.high.y * 0.5, box.low.z * 0.5 + box.high.z * 0.5);
}

inline __device__ float longestSideOfAabb(const Aabb & box)
{
	float x = box.high.x - box.low.x;
	float y = box.high.y - box.low.y;
	float z = box.high.z - box.low.z;
	x = (x > y) ? x : y;
	return (z > x) ? z : x;
}

inline __device__ int isAabbOverlapping(const Aabb & a, const Aabb & b)
{
    if(a.low.x >= b.high.x) return 0;
    if(a.low.y >= b.high.y) return 0;
    if(a.low.z >= b.high.z) return 0;
    if(b.low.x >= a.high.x) return 0;
    if(b.low.y >= a.high.y) return 0;
    if(b.low.z >= a.high.z) return 0;
    return 1;
    /*return	(a.low.x < b.high.x) && (b.low.x < a.high.x) && 
            (a.low.y < b.high.y) && (b.low.y < a.high.y) && 
            (a.low.z < b.high.z) && (b.low.z < a.high.z);*/
}

inline __device__ float spanOfAabb(Aabb * box, uint dimension)
{
    return (float3_component(box->high, dimension) 
            - float3_component(box->low, dimension));
}

inline __device__ float areaOfAabb(Aabb * box)
{
    float dx = box->high.x - box->low.x;
    float dy = box->high.y - box->low.y;
    float dz = box->high.z - box->low.z;
    if(dx <= 0.f || dy <= 0.f || dz <= 0.f) return 0.f;
    return (dx * dy + dy * dz + dz * dx) * 2.f;
}

inline __device__ float lowPlaneOfAabb(Aabb * box, int dimension)
{
    return float3_component(box->low, dimension);
}

inline __device__ float highPlaneOfAabb(Aabb * box, int dimension)
{
    return float3_component(box->high, dimension);
}

#endif        //  #ifndef AABB_CUH

