#ifndef RAY_INTERSECTION_CUH
#define RAY_INTERSECTION_CUH

#include "VectorMath.cuh"

struct __align__(16) Ray4 {
	float4 o;	// origin
	float4 d;	// direction
};

struct __align__(4) Aabb {
    float3 low;
    float3 high;
};

struct __align__(16) Aabb4 {
    float4 low;
    float4 high;
};

__constant__ float3 c_ray_box_face[6];

template<typename T>
inline __device__ void aabb4_convert(Aabb4 & v, const T & src)
{ 
    v3_convert<float4, float3>(v.low, src.low); 
    v3_convert<float4, float3>(v.high, src.high); 
}

inline __device__ void aabb4_reset(Aabb4 & b)
{
    b.low.x = b.low.y = b.low.z = 1e20f;
    b.high.x = b.high.y = b.high.z = -1e20f;
}

template<typename T>
inline __device__ void aabb4_expand(Aabb4 & b, 
                                    const T & pnt)
{
    if(b.low.x > pnt.x) b.low.x = pnt.x;
    if(b.low.y > pnt.y) b.low.y = pnt.y;
    if(b.low.z > pnt.z) b.low.z = pnt.z;
    if(b.high.x < pnt.x) b.high.x = pnt.x;
    if(b.high.y < pnt.y) b.high.y = pnt.y;
    if(b.high.z < pnt.z) b.high.z = pnt.z;
}

inline __device__ void aabb4_split(Aabb4 & lftBox, Aabb4 & rgtBox, 
                                const Aabb4 & box, 
                                const int & axis, 
                                const float & splitPos)
{
    lftBox = box;
    rgtBox = box;
    if(axis == 0) {
        lftBox.high.x = splitPos;
        rgtBox.low.x = splitPos;
    }
    else if(axis == 1) {
        lftBox.high.y = splitPos;
        rgtBox.low.y = splitPos;
    }
    else {
        lftBox.high.z = splitPos;
        rgtBox.low.z = splitPos;
    }
}

inline __device__ int aabb4_touch(const Aabb4 & a,
                                const Aabb4 & b)
{
    if(a.low.x >= b.high.x || a.high.x <= b.low.x) return 0;
	if(a.low.y >= b.high.y || a.high.y <= b.low.y) return 0;
	if(a.low.z >= b.high.z || a.high.z <= b.low.z) return 0;
	return 1;
}

template<typename Tbox, typename Tpnt>
inline __device__ int is_v3_inside(const Tbox & box, 
                                    const Tpnt & p)
{
    if(p.x < box.low.x || p.x > box.high.x) return 0;
	if(p.y < box.low.y || p.y > box.high.y) return 0;
	if(p.z < box.low.z || p.z > box.high.z) return 0;
	return 1;
}

inline __device__ int is_approximate(const float & a, const float & b)
{ return absoluteValueF(a-b) < 1e-5f; }

inline __device__ int side_on_aabb4(const Aabb4 & b,
                                    const float3 & pnt)
{
    if(b.low.x - pnt.x  > 1e-5f ) return 0;
    if(pnt.x - b.high.x > 1e-5f ) return 1;
    if(b.low.y - pnt.y  > 1e-5f ) return 2;
    if(pnt.y - b.high.y > 1e-5f ) return 3;
    if(b.low.z - pnt.z  > 1e-5f ) return 4;
    return 5;    
}

inline __device__ void ray_progress(float3 & p, const Ray4 & r, float h)
{ 
  p.x = r.o.x + r.d.x * h;
  p.y = r.o.y + r.d.y * h;
  p.z = r.o.z + r.d.z * h; 
}

inline __device__ void weightedSum(float3 & dst,
                            float3 & v1,
                            float3 & v2,
                            float3 & v3,
                            float w1,
                            float w2,
                            float w3)
{
    dst.x = v1.x * w1 + v2.x * w2 + v3.x * w3;
    dst.y = v1.y * w1 + v2.y * w2 + v3.y * w3;
    dst.z = v1.z * w1 + v2.z * w2 + v3.z * w3;
}

inline __device__ int ray_box(const Ray4 & ray,
                        const Aabb4 & aabb,
                        float & tmin, float & tmax)
{
	//AABB is considered as 3 pairs of 2 planes( {x_min, x_max}, {y_min, y_max}, {z_min, z_max} ).
	//t_min is the point of intersection with the closer plane, t_max is the point of intersection with the farther plane.
	//
	//if (rayNormalizedDirection.x < 0.0f), then max.x will be the near plane 
	//and min.x will be the far plane; otherwise, it is reversed.
	//
	//In order for there to be a collision, the t_min and t_max of each pair must overlap.
	//This can be tested for by selecting the highest t_min and lowest t_max and comparing them.
	
	//int3 isNegative = isless( rayNormalizedDirection, make_float3(0.0f, 0.0f, 0.0f) );	//isless(x,y) returns (x < y)
	int3 isNegative = make_int3(ray.d.x < 0.f, ray.d.y < 0.f, ray.d.z < 0.f);
	//When using vector types, the select() function checks the most signficant bit, 
	//but isless() sets the least significant bit.
	//isNegative <<= 31;

	//select(b, a, condition) == condition ? a : b
	//When using select() with vector types, (condition[i]) is true if its most significant bit is 1
	float3 t_min = float4_difference( select4(aabb.high, aabb.low, isNegative), ray.o );
	float3 t_max = float4_difference( select4(aabb.low, aabb.high, isNegative), ray.o );
	
	v3_divide_inplace<float3, float4>(t_min, ray.d);
	v3_divide_inplace<float3, float4>(t_max, ray.d);
	
	//Must use fmin()/fmax(); if one of the parameters is NaN, then the parameter that is not NaN is returned. 
	//Behavior of min()/max() with NaNs is undefined. (See OpenCL Specification 1.2 [6.12.2] and [6.12.4])
	//Since the innermost fmin()/fmax() is always not NaN, this should never return NaN.
	tmin = fmax( t_min.z, fmax(t_min.y, fmax(t_min.x, ray.o.w)) );
	tmax = fmin( t_max.z, fmin(t_max.y, fmin(t_max.x, ray.d.w)) );

	if(tmin >= tmax || tmax <= 0.f) return 0;
	
	return 1;
}
#endif        //  #ifndef RAY_INTERSECTION_CUH

