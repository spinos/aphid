#ifndef _MATRIX_MATH_H_
#define _MATRIX_MATH_H_

#include "bvh_common.h"

inline __device__ void fill_mat44(mat44 & m, float3 a, float3 b, float3 c, float3 d)
{
    m.v[0] = make_float4(a.x, a.y, a.z, 1.0f);
    m.v[1] = make_float4(b.x, b.y, b.z, 1.0f);
    m.v[2] = make_float4(c.x, c.y, c.z, 1.0f);
    m.v[3] = make_float4(d.x, d.y, d.z, 1.0f);
}

inline __device__ float determinant33( float a, float b, float c, float d, float e, float f, float g, float h, float i )
{
    return ( a*( e*i - h*f ) - b*( d*i - g*f ) + c*( d*h - g*e ) );
}

inline __device__ float determinant44(mat44 M)
{
    return  ( M.v[0].x * determinant33( M.v[1].y, M.v[2].y, M.v[3].y, M.v[1].z, M.v[2].z, M.v[3].z, M.v[1].w, M.v[2].w, M.v[3].w )
			- M.v[1].x * determinant33( M.v[0].y, M.v[2].y, M.v[3].y, M.v[0].z, M.v[2].z, M.v[3].z, M.v[0].w, M.v[2].w, M.v[3].w )
			+ M.v[2].x * determinant33( M.v[0].y, M.v[1].y, M.v[3].y, M.v[0].z, M.v[1].z, M.v[3].z, M.v[0].w, M.v[1].w, M.v[3].w )
			- M.v[3].x * determinant33( M.v[0].y, M.v[1].y, M.v[2].y, M.v[0].z, M.v[1].z, M.v[2].z, M.v[0].w, M.v[1].w, M.v[2].w ) );

}

#endif
