#ifndef RAY_INTERSECTION_CUH
#define RAY_INTERSECTION_CUH

#include <bvh_math.cuh>

struct __align__(16) Ray {
	float4 o;	// origin
	float4 d;	// direction
};

__device__ void ray_progress(float3 & p, const Ray & r, float h)
{ 
  p.x = r.o.x + r.d.x * h;
  p.y = r.o.y + r.d.y * h;
  p.z = r.o.z + r.d.z * h; 
}

template<typename T1, typename T2>
__device__ float3 float3_cross(const T1 & v1, const T2 & v2)
{ return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x); }

template<typename T1, typename T2>
__device__ float3 float3_difference(const T1 & v1, const T2 & v2)
{ return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z); }

template<typename T1, typename T2>
__device__ float float3_dot(const T1 & v1, const T2 & v2)
{ return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z); }

__device__ void float3_divide_inplace(float3 & v1, const float4 & v2)
{ 
    v1.x /= v2.x;
    v1.y /= v2.y;
    v1.z /= v2.z; 
}

__device__ void weightedSum(float3 & dst,
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

__device__ int ray_box(float & distanceMin, float & distanceMax,
                        Ray ray,
                        float rayLength, 
                        const Aabb & aabb)
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
	float3 t_min = float3_difference( select(aabb.high, aabb.low, isNegative), ray.o );
	float3 t_max = float3_difference( select(aabb.low, aabb.high, isNegative), ray.o );
	
	float3_divide_inplace(t_min, ray.d);
	float3_divide_inplace(t_max, ray.d);
	
	distanceMin = 0.f; 
	distanceMax = rayLength;

	//Must use fmin()/fmax(); if one of the parameters is NaN, then the parameter that is not NaN is returned. 
	//Behavior of min()/max() with NaNs is undefined. (See OpenCL Specification 1.2 [6.12.2] and [6.12.4])
	//Since the innermost fmin()/fmax() is always not NaN, this should never return NaN.
	distanceMin = fmax( t_min.z, fmax(t_min.y, fmax(t_min.x, distanceMin)) );
	distanceMax = fmin( t_max.z, fmin(t_max.y, fmin(t_max.x, distanceMax)) );

	return (distanceMin < distanceMax && distanceMax > 0.f);
}
#endif        //  #ifndef RAY_INTERSECTION_CUH

