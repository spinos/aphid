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

void decompressIndices(uint * decompressedIndices,
                    uint * compressedIndices,
					KeyValuePair * sorted,
					uint * offset,
					uint * runLength,
					uint n)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(n, tpb);
    dim3 grid(nblk, 1, 1);
    
    decompressIndices_kernel<<< grid, block>>>(decompressedIndices,
                                            compressedIndices,
                                            sorted,
                                          offset,
                                          runLength,
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

void writeSortedHash(KeyValuePair * dst,
							KeyValuePair * src,
							uint * indices,
							uint n)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(n, tpb);
    dim3 grid(nblk, 1, 1);
    
    writeSortedHash_kernel<<< grid, block>>>(dst,
							src,
							indices,
							n);
}

void rearrangeIndices(KeyValuePair * dst,
                        KeyValuePair * src,
                        uint * compressedIndices,
					KeyValuePair * sorted,
					uint * offset,
					uint * runLength,
					uint nunRuns)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(nunRuns, tpb);
    dim3 grid(nblk, 1, 1);
    
    rearrangeIndices_kernel<<< grid, block>>>(dst,
                            src,
                            compressedIndices,
                            sorted,
                            offset,
                            runLength,
                            nunRuns);
}

}
