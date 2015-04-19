#ifndef CUFEMMATH_CU
#define CUFEMMATH_CU

#include "bvh_math.cu"
#include "matrix_math.cu"
#include <CudaBase.h>

#include "bvh_math.cu"
#include "matrix_math.cu"
#include <CudaBase.h>

inline __device__ void extractTetij(uint c, uint & tet, uint & i, uint & j)
{
    tet = c>>5;
    i = (c & 31)>>3;
    j = c&3;
}

inline __device__ void extractTetijt(uint c, uint & tet, uint & i, uint & j, uint & t)
{
    tet = c>>5;
    i = (c & 31)>>3;
    j = (c & 7)>>1;
	t = c & 1;
}

inline __device__ void tetrahedronP(float3 * pnt,
                                    float3 * src,
                                        const uint4 & t) 
{
    pnt[0] = src[t.x];
	pnt[1] = src[t.y];
	pnt[2] = src[t.z];
	pnt[3] = src[t.w];
}

inline __device__ void tetrahedronEdge(float3 & e1, 
                                        float3 & e2, 
                                        float3 & e3, 
                                        const float3 * p) 
{
    e1 = float3_difference(p[1], p[0]);
	e2 = float3_difference(p[2], p[0]);
	e3 = float3_difference(p[3], p[0]);
}

inline __device__ void tetrahedronEdgei(float3 & e1, 
                                        float3 & e2, 
                                        float3 & e3, 
                                        float3 * p,
                                        const uint4 & v) 
{
    e1 = float3_difference(p[v.y], p[v.x]);
	e2 = float3_difference(p[v.z], p[v.x]);
	e3 = float3_difference(p[v.w], p[v.x]);
}

inline __device__ float tetrahedronVolume(const float3 & e1, 
                                        const float3 & e2, 
                                        const float3 & e3) 
{
    return float3_dot(e1, float3_cross(e2, e3)) * .16666667f;
}

inline __device__ float tetrahedronVolume(const float3 * p) 
{
    float3 e1, e2, e3;
	tetrahedronEdge(e1, e2, e3, p); 
	return tetrahedronVolume(e1, e2, e3);
}

inline __device__ void calculateBandVolume(float3 * B,
                                    float & volume,
                                    float3 * pos,
                                    const uint4 & tetv)
{
    float3 e10, e20, e30;
	
    tetrahedronEdgei(e10, e20, e30, pos, tetv);
    
    float invDetE = 1.f / determinant33(e10.x, e10.y, e10.z,
                                e20.x, e20.y, e20.z,
                                e30.x, e30.y, e30.z);
    
    B[1].x = (e20.z*e30.y - e20.y*e30.z)*invDetE;
    B[2].x = (e10.y*e30.z - e10.z*e30.y)*invDetE;
    B[3].x = (e10.z*e20.y - e10.y*e20.z)*invDetE;
    B[0].x = -B[1].x-B[2].x-B[3].x;

    B[1].y = (e20.x*e30.z - e20.z*e30.x)*invDetE;
    B[2].y = (e10.z*e30.x - e10.x*e30.z)*invDetE;
    B[3].y = (e10.x*e20.z - e10.z*e20.x)*invDetE;
    B[0].y = -B[1].y-B[2].y-B[3].y;

    B[1].z = (e20.y*e30.x - e20.x*e30.y)*invDetE;
    B[2].z = (e10.x*e30.y - e10.y*e30.x)*invDetE;
    B[3].z = (e10.y*e20.x - e10.x*e20.y)*invDetE;
    B[0].z = -B[1].z-B[2].z-B[3].z;
    
    volume = tetrahedronVolume(e10, e20, e30);
}

inline __device__ void calculateKe(mat33 & Ke,
                                float3 * B,
                                float d16,
                                float d17,
                                float d18,
                                float volume,
                                uint i,
                                uint j)
{
    Ke.v[0].x = d16 * B[i].x * B[j].x + d18 * (B[i].y * B[j].y + B[i].z * B[j].z);
    Ke.v[0].y = d17 * B[i].x * B[j].y + d18 * (B[i].y * B[j].x);
    Ke.v[0].z = d17 * B[i].x * B[j].z + d18 * (B[i].z * B[j].x);

    Ke.v[1].x = d17 * B[i].y * B[j].x + d18 * (B[i].x * B[j].y);
    Ke.v[1].y = d16 * B[i].y * B[j].y + d18 * (B[i].x * B[j].x + B[i].z * B[j].z);
    Ke.v[1].z = d17 * B[i].y * B[j].z + d18 * (B[i].z * B[j].y);

    Ke.v[2].x = d17 * B[i].z * B[j].x + d18 * (B[i].x * B[j].z);
    Ke.v[2].y = d17 * B[i].z * B[j].y + d18 * (B[i].y * B[j].z);
    Ke.v[2].z = d16 * B[i].z * B[j].z + d18 * (B[i].y * B[j].y + B[i].x * B[j].x);
    
    mat33_mult_f(Ke, volume);
}


inline void calculateIsotropicElasticity(float & d16, float & d17, float & d18)
{
    float nu = 0.33f; 
    float Y = 500000.0f;
    
    float d15 = Y / (1.0f + nu) / (1.0f - 2 * nu);
    d16 = (1.0f - nu) * d15;
    d17 = nu * d15;
    d18 = Y / 2 / (1.0f + nu);
}

#endif        //  #ifndef CUFEMMATH_CU

