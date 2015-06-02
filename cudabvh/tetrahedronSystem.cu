#include "TetrahedronSystemInterface.h"
#include "TetrahedronSystem.cuh"

extern "C" {
void tetrahedronSystemIntegrate(float3 * o_position, float3 * i_velocity, 
                                    float dt, uint n)
{
    dim3 block(512, 1, 1);
	unsigned nblk = iDivUp(n, 512);
	dim3 grid(nblk, 1, 1);
	
	tetrahedronSystemIntegrate_kernel<<< grid, block >>>(o_position, i_velocity, dt, n);
}
}

namespace tetrasys {
    
void writeVicinity(int * vicinities,
                    int * indices,
                    int * offsets,
                    uint n)
{
    dim3 block(512, 1, 1);
	unsigned nblk = iDivUp(n, 512);
	dim3 grid(nblk, 1, 1);
	
	writeVicinity_kernel<TETRAHEDRONSYSTEM_VICINITY_LENGTH> <<< grid, block >>>(vicinities,
                      indices,
                      offsets,
                      n);
}

void formTetrahedronAabbs(Aabb *dst, 
                        float3 * pos, 
                        float3 * vel, 
                        float timeStep, 
                        uint4 * tets, 
                        unsigned numTetrahedrons)
{
    int tpb = CALC_TETRA_AABB_NUM_THREADS;

    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numTetrahedrons<<2, tpb);
    
    dim3 grid(nblk, 1, 1);
    formTetrahedronAabbs_kernel<<< grid, block >>>(dst, pos, vel, timeStep, tets, numTetrahedrons<<2);
}

}

