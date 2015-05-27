#ifndef BVH_MATH_CUH
#define BVH_MATH_CUH

#include "bvh_common.h"

template<typename T>
inline __device__ void ascentOrder(T & in)
{ 
    if(in.x > in.y) {
        T c;
        c.x = in.x;
        in.x = in.y;
        in.y = c.x;
    }
}

inline __device__ float absoluteValueF(float a)
{ return (a > 0.f) ? a : -a; }

inline __device__ int isLeafNode(int index) 
{ return (index >> 31 == 0); }

inline __device__ int getIndexWithInternalNodeMarkerSet(int isLeaf, int index) 
{ return (isLeaf) ? index : (index | 0x80000000); }

inline __device__ int getIndexWithInternalNodeMarkerRemoved(int index) 
{ return index & (~0x80000000); }

inline __device__ float3 float3_from_float4(const float4 & a)
{ return make_float3(a.x, a.y, a.z); }

inline __device__ void float3_set_zero(float3 & v)
{ v.x = v.y = v.z = 0.f; }

inline __device__ float float3_component(float3 & v, int d)
{ if(d < 1) return v.x;
    if(d<2) return v.y;
    return v.z;    
}

inline __device__ float float3_length(const float3 & v) 
{ return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); }

inline __device__ float float4_length(const float4 & v) 
{ return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); }

inline __device__ float float3_length2(const float3 & v) 
{ return (v.x*v.x + v.y*v.y + v.z*v.z); }

inline __device__ float3 float3_normalize(const float3 & v)
{
	float l = float3_length(v);
	l = 1.0 / l;
	return make_float3(v.x * l, v.y * l, v.z * l);
}

inline __device__ float3 float3_reverse(const float3 & v)
{ return make_float3(-v.x, -v.y, -v.z); }

inline __device__ float3 float3_difference(const float3 & v1, const float3 & v2)
{ return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z); }

inline __device__ float3 float3_add(const float3 & v1, const float3 & v2)
{ return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z); }

inline __device__ void float3_add_inplace(float3 & v1, const float3 & v2)
{ 
    v1.x += v2.x;
    v1.y += v2.y;
    v1.z += v2.z; 
}

inline __device__ void float3_minus_inplace(float3 & v1, const float3 & v2)
{ 
    v1.x -= v2.x;
    v1.y -= v2.y;
    v1.z -= v2.z; 
}

inline __device__ void float3_scale_inplace(float3 & v1, float s)
{ 
    v1.x *= s;
    v1.y *= s;
    v1.z *= s; 
}

inline __device__ void float3_divide_inplace(float3 & v1, float s)
{ 
    v1.x /= s;
    v1.y /= s;
    v1.z /= s; 
}

inline __device__ float3 float3_progress(const float3 & p0, const float3 & v0, float h)
{ return make_float3(p0.x + v0.x * h, p0.y + v0.y * h, p0.z + v0.z * h); }

inline __device__ float3 scale_float3_by(const float3 & v, float s)
{ return make_float3(v.x * s, v.y * s, v.z * s); }

inline __device__ float distance_between(const float3 & v1, const float3 & v0)
{
    return float3_length(float3_difference(v1, v0)); 
}

inline __device__ float distance2_between(const float3 & v1, const float3 & v0)
{
    return float3_length2(float3_difference(v1, v0)); 
}

inline __device__ void float3_average4(float3 & dst, float3 * v, uint4 i)
{
    dst = v[i.x];
	dst = float3_add(dst, v[i.y]);
	dst = float3_add(dst, v[i.z]);
	dst = float3_add(dst, v[i.w]);
	dst = scale_float3_by(dst, 0.25f);
}

inline __device__ void float3_average4_direct(float3 & dst, float3 * v)
{
    dst = v[0];
	dst = float3_add(dst, v[1]);
	dst = float3_add(dst, v[2]);
	dst = float3_add(dst, v[3]);
	dst = scale_float3_by(dst, 0.25f);
}

inline __device__ int3 isless(float3 v, float3 threshold)
{ return make_int3(v.x < threshold.x, v.y < threshold.y, v.z < threshold.z); }

inline __device__ float3 select(float3 a, float3 b, int3 con)
{
    float x = con.x ? a.x : b.x;
    float y = con.y ? a.y : b.y;
    float z = con.z ? a.z : b.z;
    return make_float3(x, y, z);
}

inline __device__ float float3_dot(float3 v1, float3 v0)
{ return (v1.x * v0.x + v1.y * v0.y + v1.z * v0.z); }

inline __device__ float3 float3_cross(float3 v1, float3 v2)
{ return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x); }

inline __device__ void float3_cross1(float3 & result, float3 & v1, float3 & v2)
{ 
    result.x = (v1.y * v2.z - v1.z * v2.y);
    result.y = (v1.z * v2.x - v1.x * v2.z);
    result.z = (v1.x * v2.y - v1.y * v2.x); 
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
	if(b.low.x < dst.low.x) dst.low.x = b.low.x;
    if(b.low.y < dst.low.y) dst.low.y = b.low.y;
    if(b.low.z < dst.low.z) dst.low.z = b.low.z;
    if(b.high.x > dst.high.x) dst.high.x = b.high.x;
    if(b.high.y > dst.high.y) dst.high.y = b.high.y;
    if(b.high.z > dst.high.z) dst.high.z = b.high.z;
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
    return	(a.low.x < b.high.x) && (b.low.x < a.high.x) && 
            (a.low.y < b.high.y) && (b.low.y < a.high.y) && 
            (a.low.z < b.high.z) && (b.low.z < a.high.z);
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

#endif        //  #ifndef BVH_MATH_CUH

