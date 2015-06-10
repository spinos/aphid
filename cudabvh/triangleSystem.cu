#include "TriangleSystem.cuh"

namespace trianglesys {
void formTetrahedronAabbs(Aabb * dst,
                                float3 * pos,
                                float3 * vel,
                                float timeStep,
                                uint4 * tets,
                                uint numTriangles)
{
    int tpb = CALC_TETRA_AABB_NUM_THREADS;

    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numTriangles<<2, tpb);
    
    dim3 grid(nblk, 1, 1);
    formTriangleAabbs_kernel<<< grid, block >>>(dst, pos, vel, timeStep, tets, numTriangles<<2);
}

}
