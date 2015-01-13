#ifndef _BVH_MATH_H_
#define _BVH_MATH_H_
#include "bvh_common.h"
inline __device__ int isLeafNode(int index) 
{ return (index >> 31 == 0); }

inline __device__ int getIndexWithInternalNodeMarkerSet(int isLeaf, int index) 
{ return (isLeaf) ? index : (index | 0x80000000); }

inline __device__ int getIndexWithInternalNodeMarkerRemoved(int index) 
{ return index & (~0x80000000); }

inline __device__ float float3_length(float3 v) 
{ return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); }

inline __device__ float float3_length2(float3 v) 
{ return (v.x*v.x + v.y*v.y + v.z*v.z); }

inline __device__ float3 float3_normalize(float3 v)
{
	float l = float3_length(v);
	l = 1.0 / l;
	return make_float3(v.x * l, v.y * l, v.z * l);
}

inline __device__ float3 float3_difference(float3 v1, float3 v0)
{ return make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z); }

inline __device__ float3 float3_add(float3 v1, float3 v0)
{ return make_float3(v1.x + v0.x, v1.y + v0.y, v1.z + v0.z); }

inline __device__ float3 scale_float3_by(float3 & v, float s)
{ return make_float3(v.x * s, v.y * s, v.z * s); }

inline __device__ int3 isless(float3 v, float3 threshold)
{ return make_int3(v.x < threshold.x, v.y < threshold.y, v.z < threshold.z); }

inline __device__ float3 select(float3 a, float3 b, int3 con)
{
    float x = con.x ? a.x : b.x;
    float y = con.y ? a.y : b.y;
    float z = con.z ? a.z : b.z;
    return make_float3(x, y, z);
}

inline __device__ void resetAabb(Aabb & dst)
{
    dst.low = make_float3(HUGE_VALUE, HUGE_VALUE, HUGE_VALUE);
    dst.high = make_float3(-HUGE_VALUE, -HUGE_VALUE, -HUGE_VALUE);
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

inline __device__ void expandAabb(Aabb & dst, float4 p)
{
    if(p.x < dst.low.x) dst.low.x = p.x;
    if(p.y < dst.low.y) dst.low.y = p.y;
    if(p.z < dst.low.z) dst.low.z = p.z;
    if(p.x > dst.high.x) dst.high.x = p.x;
    if(p.y > dst.high.y) dst.high.y = p.y;
    if(p.z > dst.high.z) dst.high.z = p.z;
}

inline __device__ void expandAabb(Aabb & dst, const Aabb & b)
{
	expandAabb(dst, b.low);
	expandAabb(dst, b.high);
}

inline __device__ void expandAabb(Aabb & dst, volatile Aabb * src)
{
    if(src->low.x < dst.low.x) dst.low.x = src->low.x;
    if(src->low.y < dst.low.y) dst.low.y = src->low.y;
    if(src->low.z < dst.low.z) dst.low.z = src->low.z;
    if(src->high.x > dst.high.x) dst.high.x = src->high.x;
    if(src->high.y > dst.high.y) dst.high.y = src->high.y;
    if(src->high.z > dst.high.z) dst.high.z = src->high.z;
}

inline __device__ void copyVola(volatile Aabb * dst, const Aabb & src)
{
    dst->low.x = src.low.x;
    dst->low.y = src.low.y;
    dst->low.z = src.low.z;
    dst->high.x = src.high.x;
    dst->high.y = src.high.y;
    dst->high.z = src.high.z;
}

inline __device__ float3 centroidOfAabb(const Aabb & box)
{
	return make_float3(box.low.x * 0.5 + box.high.x * 0.5, box.low.y * 0.5 + box.high.y * 0.5, box.low.z * 0.5 + box.high.z * 0.5);
}

#endif
