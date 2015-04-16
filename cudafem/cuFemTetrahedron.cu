#include "cuFemTetrahedron_implement.h"
#include "bvh_math.cu"
#include "matrix_math.cu"
#include <CudaBase.h>

inline __device__ void tetrahedronP(float3 * pnt,
                                    float3 * src,
                                        uint4 & t) 
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

__global__ void resetRe_kernel(mat33* dst, 
                    uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	set_mat33_identity(dst[ind]);
}

__global__ void calculateRe_kernel(mat33 * dst, 
                                    float3 * pos, 
                                    float3 * pos0,
                                    uint4 * indices,
                                    uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	uint4 & t = indices[ind];
	
	float3 pnt[4];
	tetrahedronP(pnt, pos0, t);
	float3 e01, e02, e03;
	tetrahedronEdge(e01, e02, e03, pnt); 
	
	float div6V = 1.f / tetrahedronVolume(e01, e02, e03) * 6.f;

	tetrahedronP(pnt, pos, t);
	float3 e1, e2, e3;
	tetrahedronEdge(e1, e2, e3, pnt); 
	float3 n1 = scale_float3_by(float3_cross(e2, e3), div6V);
	float3 n2 = scale_float3_by(float3_cross(e3, e1), div6V);
	float3 n3 = scale_float3_by(float3_cross(e1, e3), div6V);
	
	mat33 & Re = dst[ind];
	Re.v[0].x = e01.x * n1.x + e02.x * n2.x + e03.x * n3.x;  
	Re.v[1].x = e01.x * n1.y + e02.x * n2.y + e03.x * n3.y;   
	Re.v[2].x = e01.x * n1.z + e02.x * n2.z + e03.x * n3.z;

    Re.v[0].y = e01.y * n1.x + e02.y * n2.x + e03.y * n3.x;  
	Re.v[1].y = e01.y * n1.y + e02.y * n2.y + e03.y * n3.y;   
	Re.v[2].y = e01.y * n1.z + e02.y * n2.z + e03.y * n3.z;

    Re.v[0].z = e01.z * n1.x + e02.z * n2.x + e03.z * n3.x;  
	Re.v[1].z = e01.z * n1.y + e02.z * n2.y + e03.z * n3.y;  
	Re.v[2].z = e01.z * n1.z + e02.z * n2.z + e03.z * n3.z;
	
	mat33_orthoNormalize(Re);
}

extern "C" {
void cuFemTetrahedron_resetRe(mat33 * d, uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    resetRe_kernel<<< grid, block >>>(d, maxInd);
}

void cuFemTetrahedron_calculateRe(mat33 * dst, 
                                    float3 * pos, 
                                    float3 * pos0,
                                    uint4 * indices,
                                    uint maxInd)
{
    int tpb = CudaBase::LimitNThreadPerBlock(24, 50);
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(maxInd, tpb);
    dim3 grid(nblk, 1, 1);
    
    calculateRe_kernel<<< grid, block >>>(dst, 
                                       pos, 
                                       pos0,
                                       indices,
                                       maxInd);
}

}
