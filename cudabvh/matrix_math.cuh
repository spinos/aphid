#ifndef MATRIX_MATH_CUH
#define MATRIX_MATH_CUH
#include "bvh_common.h"

__device__ float float4_dot(const float4 &v0, const float4 &v1)
{
    return (v1.x * v0.x + v1.y * v0.y + v1.z * v0.z + v1.w * v0.w);
}

__device__ void normalize(float4 &v0)
{
    float l = sqrt(v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w);
    v0.x /= l;
    v0.y /= l;
    v0.z /= l;
    v0.w /= l;
}

__device__ float4 transform(const mat44 &M, const float4 &v)
{
    float4 r;
    r.x = float4_dot(v, M.v[0]);
    r.y = float4_dot(v, M.v[1]);
    r.z = float4_dot(v, M.v[2]);
    r.w = 1.f;
    return r;
}
#endif        //  #ifndef MATRIX_MATH_CUH

