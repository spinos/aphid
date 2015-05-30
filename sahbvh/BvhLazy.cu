#include "BvhLazy.cuh"
#include <iostream>

namespace bvhlazy {

void updateNodeAabbAtLevel(Aabb * nodeAabbs,
        int * nodeLevels,
        int2 * nodes,
        KeyValuePair * primitiveIndirections,
        Aabb * primitiveAabbs,
        int level,
	    uint n)
{
    const int tpb = 256;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(n, tpb);
    dim3 grid(nblk, 1, 1);
    
    updateNodeAabbAtLevel_kernel<<< grid, block>>>(nodeAabbs,
                            nodeLevels,
                            nodes,
                            primitiveIndirections,
                            primitiveAabbs,
                            level,
                            n);
}

}
