#include "cuConjugateGradient_implement.h"
#include <matrix_math.cu>

__global__ void addX_kernel(float3 * X,
                            float3 * residual,
                            float * d,
                            float3 * prev,
                            float3 * update,
                            float d3,
                            uint * fixed,
                            uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	d[ind] = 0.f;
	
	if(fixed[ind] > 0) return;
	
	float3_add_inplace(X[ind], scale_float3_by(prev[ind], d3));
	float3_minus_inplace(residual[ind], scale_float3_by(update[ind], d3));
	d[ind] = float3_dot(residual[ind], residual[ind]);
}

__global__ void prevresidual_kernel(float3 * prev,
                            float3 * residual,
                            mat33 * A,
                            uint * rowPtr,
                            uint * colInd,
                            uint * fixed,
                            float3 * guess,
                            float3 * rhs,
                            uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	float3_set_zero(prev[ind]);
	
	if(fixed[ind] > 0) return;
	
	residual[ind] = rhs[ind];
	
	uint j;
	float3 tmp;
	uint cur = rowPtr[ind];
	uint nextRow = rowPtr[ind+1];
	for(;cur<nextRow; cur++) {
	    j = colInd[cur];
	    mat33_float3_prod(tmp, A[cur], guess[j]);
	    float3_minus_inplace(residual[ind], tmp);
	}
	
	prev[ind] = residual[ind];
}

__global__ void addResidual_kernel(float3 * prev,
                            float3 * residual,
                            float d4,
                            uint * fixed,
                            uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	if(fixed[ind] > 0) return;
	
	float3_scale_inplace(prev[ind], d4);
	float3_add_inplace(prev[ind], residual[ind]);
}

__global__ void Ax_kernel(float3 * prev,
                            float3 * update,
                            float3 * residual,
                            float * d,
                            float * d2,
                            mat33 * A,
                            uint * rowPtr,
                            uint * colInd,
                            uint * fixed,
                            uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	float3_set_zero(update[ind]);
	d[ind] = 0.f;
	d2[ind] = 0.f;
	
	if(fixed[ind] > 0) return;
	
	uint j;
	float3 tmp;
	int cur = rowPtr[ind];
	const int nextRow = rowPtr[ind+1];
	for(;cur<nextRow; cur++) {
	    j = colInd[cur];
	    mat33_float3_prod(tmp, A[cur], prev[j]);
	    float3_add_inplace(update[ind], tmp);
	}
	d[ind] = float3_dot(residual[ind], residual[ind]);
	d2[ind] = float3_dot(prev[ind], update[ind]);
}

extern "C" {
void cuConjugateGradient_Ax(float3 * X,
                            float3 * update,
                            float3 * residual,
                            float * d,
                            float * d2,
                            mat33 * A,
                            uint * rowPtr,
                            uint * colInd,
                            uint * fixed,
                            uint maxInd)
{
    dim3 block(256, 1, 1);
    unsigned nblk = iDivUp(maxInd, 256);
    dim3 grid(nblk, 1, 1);
    
    Ax_kernel<<< grid, block >>>(X, 
                                update,
                                residual,
                                d,
                                d2,
                                A,
                                rowPtr,
                                colInd,
                                fixed,
                                maxInd);
}

void cuConjugateGradient_addResidual(float3 * prev,
                            float3 * residual,
                            float d4,
                            uint * fixed,
                            uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    addResidual_kernel<<< grid, block >>>(prev, 
                                residual,
                                d4,
                                fixed,
                                maxInd);
}

void cuConjugateGradient_prevresidual(float3 * prev,
                            float3 * residual,
                            mat33 * A,
                            uint * rowPtr,
                            uint * colInd,
                            uint * fixed,
                            float3 * guess,
                            float3 * rhs,
                            uint maxInd)
{
    dim3 block(256, 1, 1);
    unsigned nblk = iDivUp(maxInd, 256);
    dim3 grid(nblk, 1, 1);
    prevresidual_kernel<<< grid, block >>>(prev, 
                                residual,
                                A,
                                rowPtr,
                                colInd,
                                fixed,
                                guess,
                                rhs,
                                maxInd);
}

void cuConjugateGradient_addX(float3 * X,
                            float3 * residual,
                            float * d,
                            float3 * prev,
                            float3 * update,
                            float d3,
                            uint * fixed,
                            uint maxInd)
{
    dim3 block(256, 1, 1);
    unsigned nblk = iDivUp(maxInd, 256);
    dim3 grid(nblk, 1, 1);
    
    addX_kernel<<< grid, block >>>(X, 
                                residual,
                                d,
                                prev,
                                update,
                                d3,
                                fixed,
                                maxInd);
}

}
