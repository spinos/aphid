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

inline __device__ int is_approximate(const float & a, const float & b)
{ return absoluteValueF(a-b) < 1e-5f; }

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

inline __device__ int ray_box(Ray4 & ray,
                        float3 & hitP, float3 & hitN,
                        const Aabb4 & aabb)
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
	ray.o.w = fmax( t_min.z, fmax(t_min.y, fmax(t_min.x, ray.o.w)) );
	ray.d.w = fmin( t_max.z, fmin(t_max.y, fmin(t_max.x, ray.d.w)) );

	if(ray.o.w >= ray.d.w || ray.d.w <= 0.f) return 0;
	
	ray_progress(hitP, ray, ray.o.w);
	
	if(is_approximate(hitP.x, aabb.low.x) ) hitN = c_ray_box_face[0];
	if(is_approximate(hitP.x, aabb.high.x) ) hitN = c_ray_box_face[1];
	if(is_approximate(hitP.y, aabb.low.y) ) hitN = c_ray_box_face[2];
	if(is_approximate(hitP.y, aabb.high.y) ) hitN = c_ray_box_face[3];
	if(is_approximate(hitP.z, aabb.low.z) ) hitN = c_ray_box_face[4];
	if(is_approximate(hitP.z, aabb.high.z) ) hitN = c_ray_box_face[5];
	
	return 1;
}
#endif        //  #ifndef RAY_INTERSECTION_CUH

