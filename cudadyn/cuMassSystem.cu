#include <bvh_common.h>
#include "bvh_math.cuh"

__global__ void computeMass_kernel(float * dst,
                float * mass0,
                uint * anchored,
                float scale,
                uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	float m0 = mass0[ind];
	if(anchored[ind] == 0) dst[ind] = m0 * scale;
}

__global__ void useAllAnchoredVelocity_kernel(float3 * vel,
                                float3 * anchoredVel,
								uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
    float3 va = anchoredVel[ind];
    vel[ind] = va;
}

__global__ void useAnchoredVelocity_kernel(float3 * vel,
                                float3 * anchoredVel,
								uint * anchored,
								uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
    float3 va = anchoredVel[ind];
    if(anchored[ind] > 0) vel[ind] = va;
}

__global__ void integrate_kernel(float3 * pos, 
								float3 * vel,
                                float3 * vela,
								float dt, 
								uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
    float3 anchoredVel = vela[ind];
    vel[ind] = anchoredVel;
	float3_add_inplace(pos[ind], scale_float3_by(anchoredVel, dt));
}

__global__ void updatePosition_kernel(float3 * pos, 
                                float3 * pos0,
								float3 * vel,
                                float3 * anchoredVel,
								uint * anchor,
								float dt, 
								uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
    float3 va = anchoredVel[ind];
    if(anchor[ind] > 0) vel[ind] = va;
	else va = vel[ind];
	pos0[ind] = pos[ind];
	float3_add_inplace(pos[ind], scale_float3_by(va, dt));
}

__global__ void integrate2_kernel(float3 * pos, 
								float3 * vel,
                                float dt, 
								uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
    float3 va = vel[ind];
	float3_add_inplace(pos[ind], scale_float3_by(va, dt));
}

__global__ void impulseForce_kernel(float3 * force,
                           float3 * deltaVel,
                           float * mass,
                           float dt,
                           uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
/*
 *   F = J / dt
 *   J = m * dv
 */
    float m = mass[ind];
    if(m > 1e5f) force[ind] = make_float3(0.f, 0.f, 0.f);
    else force[ind] = scale_float3_by(deltaVel[ind], m / dt);
}

__global__ void computeEnergy_kernel(float * energy,
                float * mass,
                float3 * vel,
                float defaultNodeMass,
                uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	float m = mass[ind];
    if(m > 1e5f) m = defaultNodeMass;
    energy[ind] = float3_length2(vel[ind]) * m;
}

__global__ void computeLength_kernel(float * energy,
                float3 * vel,
                uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	energy[ind] = float3_length2(vel[ind]);
}

__global__ void zeroVelocity_kernel(float3 * vel,
                uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	vel[ind] = make_float3(0.f, 0.f, 0.f);
}

__global__ void setVelocity_kernel(float3 * deltaVel,
                float * mass,
                float x, float y, float z,
                uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	float m = mass[ind];
    if(m > 1e5f) deltaVel[ind] = make_float3(0.f, 0.f, 0.f);
    else deltaVel[ind] = make_float3(x, y, z);
}

namespace masssystem {
void computeMass(float * dst,
                float * mass0,
                uint * anchored,
                float scale,
                uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    computeMass_kernel<<< grid, block >>>(dst,
        mass0,
        anchored,
        scale,
        maxInd);
}

void useAnchoredVelocity(float3 * vel, 
                float3 * anchoredVel,
                uint * anchored,
                uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    useAnchoredVelocity_kernel<<< grid, block >>>(vel,
        anchoredVel,
        anchored,
        maxInd);
}

void useAllAnchoredVelocity(float3 * vel, 
                float3 * anchoredVel,
                uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    useAllAnchoredVelocity_kernel<<< grid, block >>>(vel,
        anchoredVel,
        maxInd);
}

void integrate(float3 * pos, 
                float3 * prePos,
								float3 * vel, 
                                float3 * anchoredVel,
								uint * anchor,
								float dt, 
								uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    updatePosition_kernel<<< grid, block >>>(pos,
        prePos,
        vel,
        anchoredVel,
        anchor,
        dt,
        maxInd);
}

void integrateAllAnchored(float3 * pos,
                    float3 * vel,
                    float3 * vela,
                    float dt,
                    uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    integrate_kernel<<< grid, block >>>(pos,
        vel,
        vela,
        dt,
        maxInd);
}

void integrateSimple(float3 * pos, 
                float3 * vel, 
                float dt, 
                uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    integrate2_kernel<<< grid, block >>>(pos,
        vel,
        dt,
        maxInd);
}

void impulseForce(float3 * force,
                           float3 * deltaVel,
                           float * mass,
                           float dt,
                           uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    impulseForce_kernel<<< grid, block >>>(force,
                                           deltaVel,
        mass,
        dt,
        maxInd);
}

void computeEnergy(float * dst,
                float * mass,
                float3 * vel,
                float defaultNodeMass,
                uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    computeEnergy_kernel<<< grid, block >>>(dst,
        mass,
        vel,
        defaultNodeMass,
        maxInd);
}

void computeLength(float * dst,
                float3 * vel,
                uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    computeLength_kernel<<< grid, block >>>(dst,
        vel,
        maxInd);
}


void zeroVelocity(float3 * vel,
                uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    zeroVelocity_kernel<<< grid, block >>>(vel,
        maxInd);
}

void setVelocity(float3 * deltaVel,
                float * mass,
                float x, float y, float z,
                uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    setVelocity_kernel<<< grid, block >>>(deltaVel,
        mass,
        x, y, z,
        maxInd);
}

}
