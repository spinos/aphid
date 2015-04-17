#include "cuFemTetrahedron_implement.h"
#include "bvh_math.cu"
#include "matrix_math.cu"
#include <CudaBase.h>

inline __device__ void extractTetij(uint c, uint & tet, uint & i, uint & j)
{
    tet = c>>5;
    i = (c & 31)>>3;
    j = c&3;
}

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

inline __device__ void tetrahedronEdgei(float3 & e1, 
                                        float3 & e2, 
                                        float3 & e3, 
                                        float3 * p,
                                        uint4 & v) 
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

__global__ void resetStiffnessMatrix_kernel(mat33* dst, 
                    uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	set_mat33_zero(dst[ind]);
}

__global__ void stiffnessAssembly_kernel(mat33 * dst,
                                        float d16, float d17, float d18,
                                        float3 * pos,
                                        uint4 * tetv,
                                        mat33 * orientation,
                                        KeyValuePair * tetraInd,
                                        uint * bufferIndices,
                                        uint maxBufferInd,
                                        uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	float3 e10, e20, e30;
	float3 B[4];
	float invDetE;
	mat33 Ke, Re, ReT, tmp, tmpT;
	uint iTet, i, j;
	uint cur = bufferIndices[ind];
	for(;;) {
	    if(tetraInd[cur].key != ind) break;
	    
	    extractTetij(tetraInd[cur].value, iTet, i, j);
	    tetrahedronEdgei(e10, e20, e30, pos, tetv[iTet]);
	    
	    invDetE = 1.f / determinant33(e10.x, e10.y, e10.z,
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
		
		Ke.v[0].x = d16 * B[i].x * B[j].x + d18 * (B[i].y * B[j].y + B[i].z * B[j].z);
		Ke.v[0].y = d17 * B[i].x * B[j].y + d18 * (B[i].y * B[j].x);
		Ke.v[0].z = d17 * B[i].x * B[j].z + d18 * (B[i].z * B[j].x);

		Ke.v[1].x = d17 * B[i].y * B[j].x + d18 * (B[i].x * B[j].y);
		Ke.v[1].y = d16 * B[i].y * B[j].y + d18 * (B[i].x * B[j].x + B[i].z * B[j].z);
		Ke.v[1].z = d17 * B[i].y * B[j].z + d18 * (B[i].z * B[j].y);

		Ke.v[2].x = d17 * B[i].z * B[j].x + d18 * (B[i].x * B[j].z);
		Ke.v[2].y = d17 * B[i].z * B[j].y + d18 * (B[i].y * B[j].z);
		Ke.v[2].z = d16 * B[i].z * B[j].z + d18 * (B[i].y * B[j].y + B[i].x * B[i].x);
	    
		mat33_mult_f(Ke, tetrahedronVolume(e10, e20, e30));

	    Re = orientation[iTet];
	    mat33_transpose(ReT, Re);
	    
	    mat33_cpy(tmp, Re);
	    mat33_mult(tmp, Ke);
	    mat33_mult(tmp, ReT);
	    
	    mat33_add(dst[ind], tmp);
	    
	    if(j>i) {
	        mat33_transpose(tmpT, tmp);
	        mat33_add(dst[ind], tmpT);
	    }
	    
	    cur++;
	    if(cur >= maxBufferInd) break;
	}
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

void cuFemTetrahedron_resetStiffnessMatrix(mat33 * dst,
                                    uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    resetStiffnessMatrix_kernel<<< grid, block >>>(dst, 
                                        maxInd);
}

void cuFemTetrahedron_stiffnessAssembly(mat33 * dst,
                                        float3 * pos,
                                        uint4 * vert,
                                        mat33 * orientation,
                                        KeyValuePair * tetraInd,
                                        uint * bufferIndices,
                                        uint maxBufferInd,
                                        uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    stiffnessAssembly_kernel<<< grid, block >>>(dst,
        1.f, 2.f, 3.f,
                                            pos,
                                            vert,
                                            orientation,
                                            tetraInd,
                                            bufferIndices,
                                            maxBufferInd,
                                            maxInd);
}

}
