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

void countLeaves(uint * leafLengths,
                    int * qelements,
                    int2 * nodes,
                    KeyValuePair * indirections,
                    uint * runHeads,
                    uint numHeads,
                    uint numPrimitives,
                    uint numNodes,
                    uint scanLength)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(scanLength, tpb);
    dim3 grid(nblk, 1, 1);
    
    countLeaves_kernel<<< grid, block>>>(leafLengths,
                    qelements,
                    nodes,
                    indirections,
                    runHeads,
                    numHeads,
                    numPrimitives,
                    numNodes,
                    scanLength);
}

void copyHash(KeyValuePair * dst, KeyValuePair * src,
                uint n)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(n, tpb);
    dim3 grid(nblk, 1, 1);
    
    copyHash_kernel<<< grid, block>>>(dst,
        src,
        n);
}

void decompressPrimitives(KeyValuePair * dst,
                            KeyValuePair * src,
                            int2 * nodes,
                            KeyValuePair * indirections,
                            uint* leafOffset,
                            uint * runHeads,
                            uint numHeads,
                            uint numPrimitives,
                            uint numNodes)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numNodes, tpb);
    dim3 grid(nblk, 1, 1);
    
    decompressPrimitives_kernel<<< grid, block>>>(dst,
                src,
                nodes,
                indirections,
                leafOffset,
                runHeads,
                numHeads,
                numPrimitives,
                numNodes);
}

}
