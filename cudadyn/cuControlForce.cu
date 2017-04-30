#include <bvh_common.h>
#include "bvh_math.cuh"
#include "cuPRNG.cuh"

__constant__ float3 CMASSWind;
__constant__ float3 CMASSWindU;
__constant__ float3 CMASSWindV;
__constant__ float3 CMASSGravity;

__global__ void setWind_kernel(float3 * deltaVel,
                float * mass,
                float turbulence,
                uint windSeed,
                uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	float m = mass[ind];
    if(m > 1e5f) return;
    
    unsigned sd = ind + windSeed;
    float r = HybridTaus(sd);
    
    float3 wind = make_float3(CMASSWind.x * r, CMASSWind.y * r, CMASSWind.z * r);
    
    float su = (HybridTaus(sd) - .5f) * turbulence;
    float sv = (HybridTaus(sd) - .5f) * turbulence;
    float3_add_inplace(wind, make_float3(CMASSWindU.x * su, CMASSWindU.y * su, CMASSWindU.z * su));
    float3_add_inplace(wind, make_float3(CMASSWindV.x * sv, CMASSWindV.y * sv, CMASSWindV.z * sv));

    float3_add_inplace(deltaVel[ind], wind);
}

namespace windforce {

void setWindVecs(float * u,
                float * v,
                float * w)
{ 
    cudaMemcpyToSymbol(CMASSWind, u, 12);
    cudaMemcpyToSymbol(CMASSWindU, v, 12);
    cudaMemcpyToSymbol(CMASSWindV, w, 12);
}

void setWind(float3 * deltaVel,
                float * mass,
                float turbulence,
                uint windSeed,
                uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    setWind_kernel<<< grid, block >>>(deltaVel,
        mass,
        turbulence,
        windSeed,
        maxInd);
}

}

__global__ void addGravity_kernel(float3 * deltaVel, 
								float * mass,
                                float dt, 
								uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
    if(mass[ind] < 1e5f) 
        float3_add_inplace( deltaVel[ind], scale_float3_by(CMASSGravity, dt) );
}

namespace gravityforce {
    
void setGravity(float * g)
{ cudaMemcpyToSymbol(CMASSGravity, g, 12); }

void addGravity(float3 * deltaVel,
                float * mass,
                float dt,
                uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    addGravity_kernel<<< grid, block >>>(deltaVel,
        mass,
        dt,
        maxInd);
}

}
