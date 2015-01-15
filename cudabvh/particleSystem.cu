#include "particleSystem_implement.h"

__global__ void particleSystemSimpleGravityForce_kernel(float3 * o_force, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	o_force[ind] = make_float3(0.f, -9.81f, 0.f);
}

__global__ void particleSystemIntegrate_kernel(float3 * o_position, float3 * o_velocity, 
                                    float3 * force, float dt, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	float3 A = force[ind];
	float3 u = o_velocity[ind];
	u.x += A.x * dt;
	u.y += A.y * dt;
	u.z += A.z * dt;
	float3 X = o_position[ind];
	X.x += u.x * dt;
	X.y += u.y * dt;
	X.z += u.z * dt;
	
	o_velocity[ind] = u;
	o_position[ind] = X;
}

extern "C" void particleSystemSimpleGravityForce(float3 * o_force, uint n)
{
    dim3 block(512, 1, 1);
	unsigned nblk = iDivUp(n, 512);
	dim3 grid(nblk, 1, 1);
	
	particleSystemSimpleGravityForce_kernel<<< grid, block >>>(o_force, n);
}

extern "C" void particleSystemIntegrate(float3 * o_position, float3 * o_velocity, 
                                    float3 * force, float dt, uint n)
{
    dim3 block(512, 1, 1);
	unsigned nblk = iDivUp(n, 512);
	dim3 grid(nblk, 1, 1);
	
	particleSystemIntegrate_kernel<<< grid, block >>>(o_position, o_velocity, force, dt, n);
}

