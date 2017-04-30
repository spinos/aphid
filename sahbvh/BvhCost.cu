#include "BvhCost.cuh"
#include <iostream>
namespace bvhcost {

void computeTraverseCost(float * costs,
        int2 * nodes,
        int * nodeNumPrimitives,
	    Aabb * nodeAabbs,
        uint n)
{
    const int tpb = 256;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(n, tpb);
    dim3 grid(nblk, 1, 1);
    
    computeTraverseCost_kernel<<< grid, block>>>(costs,
                            nodes,
                            nodeNumPrimitives,
                            nodeAabbs,
                            n);
}

void countPrimitviesInNodeAtLevel(int * nodeNumPrimitives,
        int * nodeLevels,
        int2 * nodes,
        int level,
	    uint n)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(n, tpb);
    dim3 grid(nblk, 1, 1);
    
    countPrimitviesInNodeAtLevel_kernel<<< grid, block>>>(nodeNumPrimitives,
        nodeLevels,
        nodes,
        level,
	    n);
}

}
