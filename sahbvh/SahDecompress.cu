#include "SahDecompress.cuh"

namespace sahdecompress {

void initHash(KeyValuePair * primitiveIndirections,
                    uint n)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(n, tpb);
    dim3 grid(nblk, 1, 1);
    
    initHash_kernel<<< grid, block>>>(primitiveIndirections,
                                        n);
}

}
