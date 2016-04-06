#ifndef VECTOR_MATH_CUH
#define VECTOR_MATH_CUH

#include "AllBase.h"

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

inline __device__ float float3_length2(const float4 & v) 
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

inline __device__ void float3_divide_inplace(float3 & v1, const float & s)
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

inline __device__ float4 select4(const float4 & a, const float4 & b, const int3 & con)
{
    float x = con.x ? a.x : b.x;
    float y = con.y ? a.y : b.y;
    float z = con.z ? a.z : b.z;
    return make_float4(x, y, z, 1.f);
}

inline __device__ float3 float4_difference(const float4 & v1, const float4 & v2)
{ return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z); }

template<typename T1, typename T2>
inline __device__ void v3_convert(T1 & a, const T2 & b)
{
    a.x = b.x;
    a.y = b.y;
    a.z = b.z;
}

template<typename T1, typename T2, typename T3>
inline __device__ void v3_add_mult(T1 & a, const T2 & b, const T3 & c)
{
    a.x += b.x * c;
    a.y += b.y * c;
    a.z += b.z * c;
}

template<typename T1, typename T2>
inline __device__ void v3_add(T1 & a, const T2 & b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

template<typename T1, typename T2, typename T3>
inline __device__ void v3_minus_mult(T1 & a, const T2 & b, const T3 & c)
{
    a.x -= b.x * c;
    a.y -= b.y * c;
    a.z -= b.z * c;
}

template<typename T1, typename T2>
inline __device__ void v3_minus(T1 & a, const T2 & b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

template<typename T>
inline __device__ void v3_normalize_inplace(T & v)
{
	float l = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
	v.x /= l;
	v.y /= l;
	v.z /= l;
}

template<typename T1, typename T2>
inline __device__ void v3_divide_inplace(T1 & v1, const T2 & v2)
{ 
    v1.x /= v2.x;
    v1.y /= v2.y;
    v1.z /= v2.z; 
}

template<typename T1, typename T2>
__device__ float3 v3_cross(const T1 & v1, const T2 & v2)
{ return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x); }

template<typename T1, typename T2>
__device__ float v3_dot(const T1 & v1, const T2 & v2)
{ return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z); }

template<typename T>
inline __device__ float v3_component(const T & v, int d)
{ 
    if(d < 1) return v.x;
    if(d<2) return v.y;
    return v.z;    
}

template<typename T>
inline __device__ void v3_reverse_inplace(T & v)
{ 
    v.x *= -1.f;
    v.y *= -1.f;
    v.z *= -1.f;
}

template<typename T>
inline __device__ int v3_major_axis(const T & v)
{ 
    float a = (v.x >= 0) ? v.x : -v.x;
	float b = (v.y >= 0) ? v.y : -v.y;
	float c = (v.z >= 0) ? v.z : -v.z;
	if(a >= b && a >= c) return 0;
	if(b >= c && b >= a) return 1;
	return 2;    
}

template<typename T>
inline __device__ int v3_orientation(const T & v)
{ 
    int j = v3_major_axis<T>(v);
    if(j==0) {
		if(v.x<0.f) return 0;
		return 1;
	}
	if(j==1) {
		if(v.y<0.f) return 2;
		return 3;
	}
	if(v.z<0.f) return 4;
	return 5;
}

template<typename T>
inline __device__ void v3_addr(T & a, const float * b)
{
    a.x += b[0];
    a.y += b[1];
    a.z += b[2];
}

template<typename T>
inline __device__ void v3_minusr(T & a, const float * b)
{
    a.x -= b[0];
    a.y -= b[1];
    a.z -= b[2];
}

template<typename T>
inline __device__ void v3_multr(T & a, const float * b)
{
    a.x *= b[0];
    a.y *= b[1];
    a.z *= b[2];
}

template<typename T>
inline __device__ void v3_divider(T & a, const float * b)
{
    a.x /= b[0];
    a.y /= b[1];
    a.z /= b[2];
}

template<typename T>
inline __device__ void v3_r(T & a, const float * b)
{
    a.x = b[0];
    a.y = b[1];
    a.z = b[2];
}

#endif        //  #ifndef VECTOR_MATH_CUH

