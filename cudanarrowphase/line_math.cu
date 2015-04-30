#ifndef LINE_MATH_CU
#define LINE_MATH_CU

#include "bvh_common.h"
#include "bvh_math.cu"

inline __device__ float distancePointLine(const float3 & P0, 
                        const float3 & P1, const float3 & P2,
                        float3 & p0p1, float3 & p2p1)
{
    p0p1 = float3_difference(P0, P1);
    p2p1 = float3_difference(P2, P1);
    return sqrt( float3_length2(float3_cross( p0p1, float3_difference(P0, P2) ) ) / 
        float3_length2(p2p1) );
}

// http://mathworld.wolfram.com/Point-PlaneDistance.html

inline __device__ float3 projectPointOnPlane(float3 p, float3 v, float3 nor)
{
    float t = float3_dot(nor, v) - float3_dot(nor, p);
    return float3_add(p, scale_float3_by(nor, t));
}

inline __device__ void computeClosestPointOnLine1(const float3 & p0, 
                                                    const float3 & p1, 
                                                    const float3 & p2,
                                                    ClosestPointTestContext & result)
{
    float3 vr = float3_difference(p0, p1);
    float dr = float3_length(vr);
	if(dr < 1e-6f) {
        result.closestPoint = p1;
		result.closestDistance = 0.f;
        return;
    }
	
    float3 v1 = float3_difference(p2, p1);
	float d1 = float3_length(v1);
	// vr = float3_normalize(vr);
	float3_scale_inplace(vr, 1.f/dr);
	// v1 = float3_normalize(v1);
	float3_scale_inplace(v1, 1.f/d1);
	float vrdv1 = float3_dot(vr, v1) * dr;
	if(vrdv1 < 0.f) vrdv1 = 0.f;
	if(vrdv1 > d1) vrdv1 = d1;
	
	v1 = float3_add(p1, scale_float3_by(v1, vrdv1));
	float dc = distance_between(v1, p0);
	
	if(dc < result.closestDistance) {
	    result.closestPoint = v1;
	    result.closestDistance = dc;
	}
}

inline __device__ void computeClosestPointOnLine(const float3 & p, const float3 * v, ClosestPointTestContext & result)
{
    computeClosestPointOnLine1(p, v[0], v[1], result);
   /*
    float3 vr = float3_difference(p, v[0]);
    float dr = float3_length(vr);
	if(dr < 1e-6f) {
        result.closestPoint = v[0];
		result.closestDistance = 0.f;
        return;
    }
	
    float3 v1 = float3_difference(v[1], v[0]);
	float d1 = float3_length(v1);
	// vr = float3_normalize(vr);
	float3_scale_inplace(vr, 1.f/dr);
	// v1 = float3_normalize(v1);
	float3_scale_inplace(v1, 1.f/d1);
	float vrdv1 = float3_dot(vr, v1) * dr;
	if(vrdv1 < 0.f) vrdv1 = 0.f;
	if(vrdv1 > d1) vrdv1 = d1;
	
	v1 = float3_add(v[0], scale_float3_by(v1, vrdv1));
	float dc = distance_between(v1, p);
	
	if(dc < result.closestDistance) {
	    result.closestPoint = v1;
	    result.closestDistance = dc;
	}
	*/
}

#endif        //  #ifndef LINE_MATH_CU

