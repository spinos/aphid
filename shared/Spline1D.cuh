#include "bvh_math.cuh"

__constant__ float CSplineCvs[8];

__inline__ __device__ float calculateBezierPoint1D(float t)
{
    float u = 1.f - t;
    float tt = t * t;
    float uu = u*u;
    float uuu = uu * u;
    float ttt = tt * t;
    
    float p = CSplineCvs[1] * uuu; //first term
    p += CSplineCvs[3] * 3.f * uu * t; //second term
    p += CSplineCvs[5] * 3.f * u * tt; //third term
    p += CSplineCvs[7] * ttt; //fourth term
    return p;
}
