#include "bvh_common.h"
#include <radixsort_implement.h>
#include "cuFemMath.cu"
#include <CudaBase.h>
#include <Spline1D.cuh>

__constant__ float3 CGravity;
__constant__ float3 CWind;

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

__global__ void elasticity_kernel(float4 * d,
                        float * alpha,
                        float Y,
                        uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	float bezier = calculateBezierPoint1D(alpha[ind]);
	if(bezier < 0.05f) bezier = 0.05f;
	
    float4 d161718;
    calculateIsotropicElasticity4(Y * bezier, 
                                     d161718);
    d[ind] = d161718;
}

__global__ void externalForce_kernel(float3 * dst,
                                float * mass,
                                float3 * velocity,
                                uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	float m = mass[ind];
	if(m > 1e28f) {
	    float3_set_zero(dst[ind]);
	    return;
	}
	
	float3 F = scale_float3_by(CGravity, m);
    
    float3 w = CWind;
    float3_scale_inplace(w, m);
    float3_add_inplace(F, w);
    
    float3 u = CWind;;
    float3_minus_inplace(u, velocity[ind]);
    float3_scale_inplace(u, m * 0.01f);
    float3_add_inplace(F, u);
    dst[ind] = F;
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
	
	float3 result;
	float3_set_zero(result);
	
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
	    float3_minus_inplace(result, tmp);
		
		mat33_mult_f(K, dt2);
		stiffness[cur] = K;
		
	    if(ind == colInd[cur]) {
	        damping = .21f * mi * dt + mi;
	        K.v[0].x += damping;
	        K.v[1].y += damping;
	        K.v[2].z += damping;
	        stiffness[cur] = K;
	    }
	}
	
	float3_minus_inplace(result, f0[ind]);
	float3_add_inplace(result, externalForce[ind]);
	float3_scale_inplace(result, dt);

	tmp = vel[ind];
	float3_scale_inplace(tmp, mi);
	float3_add_inplace(result, tmp);
	rhs[ind] = result;
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
                                    float3 * pos,
                                    uint4 * tetvert,
                                    float4 * BVol,
                                    mat33 * orientation,
                                    KeyValuePair * tetraInd,
                                    uint * bufferIndices,
                                    float4 * elasticity,
                                    uint maxBufferInd,
                                    uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	float3_set_zero(dst[ind]);
	float4 d161718;
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
		
		d161718 = elasticity[iTet];
		
	    calculateKe(Ke, B, d161718.x, d161718.y, d161718.z, i, j);	    
		
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
                                        float4 * BVol,
                                        mat33 * orientation,
                                        KeyValuePair * tetraInd,
                                        uint * bufferIndices,
                                        float4 * elasticity,
                                        uint maxBufferInd,
                                        uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	set_mat33_zero(dst[ind]);
	
    float4 d161718;
	float4 * B;
	mat33 Ke, Re, ReT, tmp, tmpT;
	uint iTet, i, j, needT;
	uint cur = bufferIndices[ind];
	for(;;) {
	    if(tetraInd[cur].key != ind) break;
	    
	    extractTetijt(tetraInd[cur].value, iTet, i, j, needT);
	    
	    B = &BVol[iTet<<2];
	    
	    d161718 = elasticity[iTet];
        
	    calculateKe(Ke, B, d161718.x, d161718.y, d161718.z, i, j);

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

}

namespace tetrahedronfem {

void setGravity(float * g)
{ cudaMemcpyToSymbol(CGravity, g, 12); }

void setWind(float * w)
{ cudaMemcpyToSymbol(CWind, w, 12); }

void computeExternalForce(float3 * dst,
                                float * mass,
                                float3 * velocity,
                                float * wind,
                                uint maxInd)
{
	setWind(wind);
    
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    externalForce_kernel<<< grid, block >>>(dst,
        mass,
        velocity,
        maxInd);
}

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
                                    float4 * elasticity,
                                    uint maxBufferInd,
                                    uint maxInd)
{
    int tpb = 256;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(maxInd, tpb);
    dim3 grid(nblk, 1, 1);
    
    internalForce_kernel<<< grid, block >>>(dst,
                                            pos,
                                            tetvert,
                                            BVol,
                                            orientation,
                                            tetraInd,
                                            bufferIndices,
                                            elasticity,
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
                                        float4 * elasticity,
                                        uint maxBufferInd,
                                        uint maxInd)
{
    int tpb = 256;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(maxInd, tpb);
    dim3 grid(nblk, 1, 1);
    
    stiffnessAssembly_kernel<<< grid, block >>>(dst,
                                            BVol,
                                            orientation,
                                            tetraInd,
                                            bufferIndices,
                                            elasticity,
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

void computeElasticity(float4 * d,
                        float * alpha,
                        float Y,
                        uint maxInd,
                        float * splineV)
{
    cudaMemcpyToSymbol(CSplineCvs, splineV, 32); 
    
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    elasticity_kernel<<< grid, block >>>(d,
        alpha,
        Y,
        maxInd);
}
}
