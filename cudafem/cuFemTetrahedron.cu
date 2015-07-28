#include "bvh_common.h"
#include <radixsort_implement.h>
#include "cuFemMath.cu"

__global__ void computeBVolume_kernel(float4 * dst, 
                    float3 * pos,
                    uint4 * tetVertices,
                    uint numTet)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= numTet) return;
	
	calculateBandVolume(&dst[ind<<2], pos, tetVertices[ind]);
}

__global__ void integrate_kernel(float3 * pos, 
								float3 * vel,
                                float3 * anchoredVel,
								uint * anchor,
								float dt, 
								uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
    float3 va = anchoredVel[ind];
	if(anchor[ind] > (1<<23)) vel[ind] = va;
	float3_add_inplace(pos[ind], scale_float3_by(vel[ind], dt));
}

__global__ void externalForce_kernel(float3 * dst,
                                float * mass,
                                uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	float m = mass[ind];
	if(m > 1e28f) {
	    float3_set_zero(dst[ind]);
	    return;
	}
	
	float3 gravity = make_float3(0.f, -9.81f, 0.f);
	dst[ind] = scale_float3_by(gravity, mass[ind]);
}

__global__ void computeRhs_kernel(float3 * rhs,
                                float3 * pos,
                                float3 * vel,
                                float * mass,
                                mat33 * stiffness,
                                uint * rowPtr,
                                uint * colInd,
                                float3 * f0,
                                float3 * externalForce,
                                float dt,
                                float dt2,
                                uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	float3_set_zero(rhs[ind]);
	
	const uint nextRow = rowPtr[ind+1];
	uint cur = rowPtr[ind];
	const float mi = mass[ind];
	mat33 K;
	uint j;
	float3 tmp;
	float damping;
	for(;cur<nextRow; cur++) {
	    K = stiffness[cur];
	    j = colInd[cur];
	    mat33_float3_prod(tmp, K, pos[j]);
	    float3_minus_inplace(rhs[ind], tmp);
		
		mat33_mult_f(stiffness[cur], dt2);
		
	    if(ind == colInd[cur]) {
	        damping = .21f * mi * dt + mi;
	        stiffness[cur].v[0].x += damping;
	        stiffness[cur].v[1].y += damping;
	        stiffness[cur].v[2].z += damping;
	    }
	}
	
	float3_minus_inplace(rhs[ind], f0[ind]);
	float3_add_inplace(rhs[ind], externalForce[ind]);
	float3_scale_inplace(rhs[ind], dt);

	tmp = vel[ind];
	float3_scale_inplace(tmp, mi);
	float3_add_inplace(rhs[ind], tmp);
}

__global__ void dampK_kernel(mat33 * stiffness,
                                float * mass,
                                uint * rowPtr,
                                uint * colInd,
                                float dt,
								float dt2,
                                uint maxInd)
{
	unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	const uint nextRow = rowPtr[ind+1];
	uint cur = rowPtr[ind];
	const float mi = mass[ind];
	float damping;
	for(;cur<nextRow; cur++) {

	    mat33_mult_f(stiffness[cur], dt2);
		
	    if(ind == colInd[cur]) {
	        damping = .08f * mi * dt + mi;
	        stiffness[cur].v[0].x += damping;
	        stiffness[cur].v[1].y += damping;
	        stiffness[cur].v[2].z += damping;
	    }
	}
}

__global__ void internalForce_kernel(float3 * dst,
                                    float * stiffnessAttenuate,
                                    float Y,
                                    float3 * pos,
                                    uint4 * tetvert,
                                    float4 * BVol,
                                    mat33 * orientation,
                                    KeyValuePair * tetraInd,
                                    uint * bufferIndices,
                                    uint maxBufferInd,
                                    uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	float3_set_zero(dst[ind]);
	float d16, d17, d18;
	float4 * B;
	float3 pj, force, sum;
	mat33 Ke, Re;
	uint iTet, i, j;
	uint cur = bufferIndices[ind];
	uint lastTet = 9496729;
	for(;;) {
	    if(tetraInd[cur].key != ind) break;
	    
	    extractTetij(tetraInd[cur].value, iTet, i, j);
	    
	    if(lastTet != iTet) {
	        if(lastTet != 9496729) {
				mat33_float3_prod(force, Re, sum);
	            float3_minus_inplace(dst[ind], force);
	        }
			
	        float3_set_zero(sum);
	        lastTet = iTet;
	    }
		
		Re = orientation[iTet];
		B = &BVol[iTet<<2];
		calculateIsotropicElasticity(Y * stiffnessAttenuate[iTet], 
                                     d16, d17, d18);
	    calculateKe(Ke, B, d16, d17, d18, i, j);	    
		
	    uint * tetv = &(tetvert[iTet].x);
	    pj = pos[tetv[j]];
		
	    mat33_float3_prod(force, Ke, pj);	    
	    float3_add_inplace(sum, force);
		
	    cur++;
	    if(cur >= maxBufferInd) break;
	}
	
	mat33_float3_prod(force, Re, sum);
	float3_minus_inplace(dst[ind], force);
}

__global__ void resetForce_kernel(float3 * dst,
    uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	float3_set_zero(dst[ind]);
}

__global__ void resetStiffnessMatrix_kernel(mat33* dst, 
                    uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	set_mat33_zero(dst[ind]);
}

__global__ void stiffnessAssembly_kernel(mat33 * dst,
                                         float * stiffnessAttenuate,
                                         float Y,
                                        float4 * BVol,
                                        mat33 * orientation,
                                        KeyValuePair * tetraInd,
                                        uint * bufferIndices,
                                        uint maxBufferInd,
                                        uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	set_mat33_zero(dst[ind]);
	
    float d16, d17, d18;
	float4 * B;
	mat33 Ke, Re, ReT, tmp, tmpT;
	uint iTet, i, j, needT;
	uint cur = bufferIndices[ind];
	for(;;) {
	    if(tetraInd[cur].key != ind) break;
	    
	    extractTetijt(tetraInd[cur].value, iTet, i, j, needT);
	    
	    B = &BVol[iTet<<2];
	    
        calculateIsotropicElasticity(Y * stiffnessAttenuate[iTet],
                                     d16, d17, d18);
	    calculateKe(Ke, B, d16, d17, d18, i, j);

	    Re = orientation[iTet];
		
		mat33_transpose(ReT, Re);
	    mat33_cpy(tmp, Re);
	    mat33_mult(tmp, Ke);
	    mat33_mult(tmp, ReT);
		mat33_transpose(tmpT, tmp);
    
	    if(needT)
	        mat33_add(dst[ind], tmpT);
		else
			mat33_add(dst[ind], tmp);
					
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
                                    float4 * BVol,
                                    uint4 * indices,
                                    uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	uint4 t = indices[ind];
	
	float4 * B = &BVol[ind<<2];
	
	float3 pnt[4];
	tetrahedronP(pnt, pos0, t);
	float3 e01, e02, e03;
	tetrahedronEdge(e01, e02, e03, pnt); 
	
	float div6V = 1.f / B[0].w * 6.f;

	tetrahedronP(pnt, pos, t);
	float3 e1, e2, e3;
	tetrahedronEdge(e1, e2, e3, pnt); 
	float3 n1 = scale_float3_by(float3_cross(e2, e3), div6V);
	float3 n2 = scale_float3_by(float3_cross(e3, e1), div6V);
	float3 n3 = scale_float3_by(float3_cross(e1, e2), div6V);
	
	mat33 * Ke = &dst[ind];
	Ke->v[0].x = e01.x * n1.x + e02.x * n2.x + e03.x * n3.x;  
	Ke->v[1].x = e01.x * n1.y + e02.x * n2.y + e03.x * n3.y;   
	Ke->v[2].x = e01.x * n1.z + e02.x * n2.z + e03.x * n3.z;

    Ke->v[0].y = e01.y * n1.x + e02.y * n2.x + e03.y * n3.x;  
	Ke->v[1].y = e01.y * n1.y + e02.y * n2.y + e03.y * n3.y;   
	Ke->v[2].y = e01.y * n1.z + e02.y * n2.z + e03.y * n3.z;

    Ke->v[0].z = e01.z * n1.x + e02.z * n2.x + e03.z * n3.x;  
	Ke->v[1].z = e01.z * n1.y + e02.z * n2.y + e03.z * n3.y;  
	Ke->v[2].z = e01.z * n1.z + e02.z * n2.z + e03.z * n3.z;
	
	mat33_orthoNormalize(*Ke);
}

extern "C" {
void cuFemTetrahedron_resetRe(mat33 * d, uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    resetRe_kernel<<< grid, block >>>(d, maxInd);
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

void cuFemTetrahedron_resetForce(float3 * dst,
                                    uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    resetForce_kernel<<< grid, block >>>(dst, maxInd);
}

void cuFemTetrahedron_computeRhs(float3 * rhs,
                                float3 * pos,
                                float3 * vel,
                                float * mass,
                                mat33 * stiffness,
                                uint * rowPtr,
                                uint * colInd,
                                float3 * f0,
                                float3 * externalForce,
                                float dt,
                                uint maxInd)
{
    int tpb = 256;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(maxInd, tpb);
    dim3 grid(nblk, 1, 1);
    
    computeRhs_kernel<<< grid, block >>>(rhs, 
        pos,
        vel,
        mass, 
        stiffness, 
        rowPtr, 
        colInd,
        f0,
        externalForce,
        dt, 
        dt * dt,
        maxInd);
}

void cuFemTetrahedron_dampK(mat33 * stiffness,
                                float * mass,
                                uint * rowPtr,
                                uint * colInd,
                                float dt,
                                uint maxInd)
{
	int tpb = CudaBase::LimitNThreadPerBlock(16, 50);
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(maxInd, tpb);
    dim3 grid(nblk, 1, 1);
    
    dampK_kernel<<< grid, block >>>(stiffness, 
        mass, 
        rowPtr, 
        colInd,
		dt,
        dt * dt,
        maxInd);
}

void cuFemTetrahedron_externalForce(float3 * dst,
                                float * mass,
                                uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    externalForce_kernel<<< grid, block >>>(dst,
        mass,
        maxInd);
}

}

namespace tetrahedronfem {
void computeBVolume(float4 * dst, 
                    float3 * pos,
                    uint4 * tetVertices,
                    uint numTet)
{
    dim3 block(256, 1, 1);
    unsigned nblk = iDivUp(numTet, 256);
    dim3 grid(nblk, 1, 1);
    
    computeBVolume_kernel<<< grid, block >>>(dst, 
                       pos,
                       tetVertices,
                       numTet);
}

void calculateRe(mat33 * dst, 
                                    float3 * pos, 
                                    float3 * pos0,
                                    uint4 * indices,
                                    float4 * BVol,
                                    uint maxInd)
{
    int tpb = 256;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(maxInd, tpb);
    dim3 grid(nblk, 1, 1);
    
    calculateRe_kernel<<< grid, block >>>(dst, 
                                       pos, 
                                       pos0,
                                       BVol,
                                       indices,
                                       maxInd);
}

void internalForce(float3 * dst,
                                    float3 * pos,
                                    uint4 * tetvert,
                                    float4 * BVol,
                                    mat33 * orientation,
                                    KeyValuePair * tetraInd,
                                    uint * bufferIndices,
                                    float * stiffnessAttenuate,
                                    float Y,
                                    uint maxBufferInd,
                                    uint maxInd)
{
    int tpb = 256;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(maxInd, tpb);
    dim3 grid(nblk, 1, 1);
    
    internalForce_kernel<<< grid, block >>>(dst,
                                            stiffnessAttenuate,
                                            Y,
                                            pos,
                                            tetvert,
                                            BVol,
                                            orientation,
                                            tetraInd,
                                            bufferIndices,
                                            maxBufferInd,
                                            maxInd);
}

void stiffnessAssembly(mat33 * dst,
                                        float3 * pos,
                                        uint4 * vert,
                                        float4 * BVol,
                                        mat33 * orientation,
                                        KeyValuePair * tetraInd,
                                        uint * bufferIndices,
                                        float * stiffnessAttenuate,
                                        float Y,
                                        uint maxBufferInd,
                                        uint maxInd)
{
    int tpb = 256;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(maxInd, tpb);
    dim3 grid(nblk, 1, 1);
    
    stiffnessAssembly_kernel<<< grid, block >>>(dst,
                                            stiffnessAttenuate,
                                            Y,
                                            BVol,
                                            orientation,
                                            tetraInd,
                                            bufferIndices,
                                            maxBufferInd,
                                            maxInd);
}

void integrate(float3 * pos, 
								float3 * vel, 
                                float3 * anchoredVel,
								uint * anchor,
								float dt, 
								uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    integrate_kernel<<< grid, block >>>(pos,
        vel,
        anchoredVel,
        anchor,
        dt,
        maxInd);
}
}
