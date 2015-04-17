#ifndef _MATRIX_MATH_H_
#define _MATRIX_MATH_H_

#include "bvh_math.cu"

inline __device__ void set_mat33_zero(mat33 & m)
{
    m.v[0] = make_float3(0.f, 0.f, 0.f);
    m.v[1] = make_float3(0.f, 0.f, 0.f);
    m.v[2] = make_float3(0.f, 0.f, 0.f);
}

inline __device__ void set_mat33_identity(mat33 & m)
{
    m.v[0] = make_float3(1.f, 0.f, 0.f);
    m.v[1] = make_float3(0.f, 1.f, 0.f);
    m.v[2] = make_float3(0.f, 0.f, 1.f);
}

inline __device__ void fill_mat44(mat44 & m, const float3 & a, const float3 & b, const float3 & c, const float3 & d)
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

inline __device__ float determinant44(const mat44 & M)
{
    return  ( M.v[0].x * determinant33( M.v[1].y, M.v[2].y, M.v[3].y, M.v[1].z, M.v[2].z, M.v[3].z, M.v[1].w, M.v[2].w, M.v[3].w )
			- M.v[1].x * determinant33( M.v[0].y, M.v[2].y, M.v[3].y, M.v[0].z, M.v[2].z, M.v[3].z, M.v[0].w, M.v[2].w, M.v[3].w )
			+ M.v[2].x * determinant33( M.v[0].y, M.v[1].y, M.v[3].y, M.v[0].z, M.v[1].z, M.v[3].z, M.v[0].w, M.v[1].w, M.v[3].w )
			- M.v[3].x * determinant33( M.v[0].y, M.v[1].y, M.v[2].y, M.v[0].z, M.v[1].z, M.v[2].z, M.v[0].w, M.v[1].w, M.v[2].w ) );

}

inline __device__ void mat33_orthoNormalize(mat33 & dst)
{
    float3 r0 = dst.v[0];
    float3 r1 = dst.v[1];
    float3 r2 = dst.v[2];
    
    float l0 = float3_length(r0);
    if(l0 > 0.f) scale_float3_by(r0, 1.f/l0);
    
    r1 = float3_difference(r1, scale_float3_by(r0, float3_dot(r0, r1)));
    
    float l1 = float3_length(r1);
    if(l1 > 0.f) scale_float3_by(r1, 1.f/l1);
    
    r2 = float3_cross(r0, r1);
    
    dst.v[0] = r0;
    dst.v[1] = r1;
    dst.v[2] = r2;
}

inline __device__ void mat33_mult_f(mat33 & m, float a)
{
    m.v[0].x *= a;
    m.v[0].y *= a;
    m.v[0].z *= a;
    m.v[1].x *= a;
    m.v[1].y *= a;
    m.v[1].z *= a;
    m.v[2].x *= a;
    m.v[2].y *= a;
    m.v[2].z *= a;
}

inline __device__ void mat33_cpy(mat33 & a, const mat33 & b)
{
    a.v[0] = b.v[0];
    a.v[1] = b.v[1];
    a.v[2] = b.v[2];
}

inline __device__ void mat33_transpose(mat33 & a, const mat33 & b)
{
    a.v[0].y = b.v[1].x;
    a.v[0].z = b.v[2].x;
    a.v[1].x = b.v[0].y;
    a.v[1].z = b.v[2].y;
    a.v[2].x = b.v[0].z;
    a.v[2].y = b.v[1].z;
}

inline __device__ void mat33_add(mat33 & a, const mat33 & b)
{
    a.v[0].x += b.v[0].x;
	a.v[0].y += b.v[0].y;
	a.v[0].z += b.v[0].z;
	a.v[1].x += b.v[1].x;
	a.v[1].y += b.v[1].y;
	a.v[1].z += b.v[1].z;
	a.v[2].x += b.v[2].x;
	a.v[2].y += b.v[2].y;
	a.v[2].z += b.v[2].z;
}

inline __device__ void mat33_mult(mat33 & a, const mat33 & b)
{
    mat33 tb;
    mat33_transpose(tb, b);
    mat33 ma;
    ma.v[0] = a.v[0];
    ma.v[1] = a.v[1];
    ma.v[2] = a.v[2];
	a.v[0].x = float3_dot(ma.v[0], tb.v[0]);
	a.v[0].y = float3_dot(ma.v[0], tb.v[1]);
	a.v[0].z = float3_dot(ma.v[0], tb.v[2]);
	a.v[1].x = float3_dot(ma.v[1], tb.v[0]);
	a.v[1].y = float3_dot(ma.v[1], tb.v[1]);
	a.v[1].z = float3_dot(ma.v[1], tb.v[2]);
	a.v[2].x = float3_dot(ma.v[2], tb.v[0]);
	a.v[2].y = float3_dot(ma.v[2], tb.v[1]);
	a.v[2].z = float3_dot(ma.v[2], tb.v[2]);
}

#endif
