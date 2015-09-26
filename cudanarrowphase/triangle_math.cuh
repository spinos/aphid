#ifndef TRIANGLE_MATH_CU
#define TRIANGLE_MATH_CU

#include "barycentric.cuh"
#include "line_math.cuh"

inline __device__ void computeClosestPointOnTriangleInside2(const float3 & p, 
                                        const float3 & v0,
                                        const float3 & v1,
                                        const float3 & v2, 
                                        ClosestPointTestContext & result)
{
    float3 nor = triangleNormal2(v0, v1, v2);
    float3 onplane = projectPointOnPlane(p, v0, nor);
    
    if(pointInsideTriangleTest2(onplane, nor, v0, v1, v2)) {
        float d = distance_between(p, onplane);
        if(d < result.closestDistance) {
            result.closestPoint = onplane;
            result.closestDistance = d;
        }
    }
}

inline __device__ void computeClosestPointOnTriangleInside(const float3 & p, const float3 * v, ClosestPointTestContext & result)
{
    computeClosestPointOnTriangleInside2(p, v[0], v[1], v[2], result);
}

inline __device__ void computeClosestPointOnTriangle2(const float3 & p, 
                                        const float3 & v0,
                                        const float3 & v1,
                                        const float3 & v2, 
                                        ClosestPointTestContext & result)
{
    float3 nor = triangleNormal2(v0, v1, v2);
    float3 onplane = projectPointOnPlane(p, v0, nor);
    
    if(pointInsideTriangleTest2(onplane, nor, v0, v1, v2)) {
        float d = distance_between(p, onplane);
        if(d < result.closestDistance) {
            result.closestPoint = onplane;
            result.closestDistance = d;
        }
        return;
    }
    
    computeClosestPointOnLine1(p, v0, v1, result);
    computeClosestPointOnLine1(p, v1, v2, result);
    computeClosestPointOnLine1(p, v2, v0, result);
}

/*
inline __device__ void computeClosestPointOnTriangle(const float3 & p, const float3 * v, ClosestPointTestContext & result)
{
    computeClosestPointOnTriangle2(p, v[0], v[1], v[2], result);
}
*/

inline __device__ void computeClosestPointOnTetrahedron(const float3 & p, const float3 * v, ClosestPointTestContext & result)
{
	computeClosestPointOnTriangleInside(p, v, result);

	float3 pr[3];
	pr[0] = v[0];
	pr[1] = v[1];
	pr[2] = v[3];
	computeClosestPointOnTriangleInside(p, pr, result);
	
	pr[0] = v[0];
	pr[1] = v[2];
	pr[2] = v[3];
	computeClosestPointOnTriangleInside(p, pr, result);
	
	pr[0] = v[1];
	pr[1] = v[2];
	pr[2] = v[3];
	computeClosestPointOnTriangleInside(p, pr, result);
	
	computeClosestPointOnLine1(p, v[0], v[1], result);
	computeClosestPointOnLine1(p, v[0], v[2], result);
	computeClosestPointOnLine1(p, v[0], v[3], result);
	computeClosestPointOnLine1(p, v[1], v[2], result);
	computeClosestPointOnLine1(p, v[2], v[3], result);
	computeClosestPointOnLine1(p, v[3], v[1], result);
}

#endif        //  #ifndef TRIANGLE_MATH_CU

