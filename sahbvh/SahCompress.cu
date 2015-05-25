#include "SahCompress.cuh"

namespace sahcompress {

void computeRunHead(uint * runHeads,
                    KeyValuePair * morton,
                    uint d,
                    uint n,
                    uint bufLength)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(bufLength, tpb);
    dim3 grid(nblk, 1, 1);
    
    computeRunHead_kernel<<< grid, block>>>(runHeads,
        morton,
        d,
        n,
        bufLength);
}

void compressRunHead(uint * compressed, 
							uint * runHeads,
							uint * indices,
							uint n)		
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(n, tpb);
    dim3 grid(nblk, 1, 1);
    
    compressRunHead_kernel<<< grid, block>>>(compressed,
        runHeads,
        indices,
        n);
}

void computeRunHash(KeyValuePair * compressed, 
						KeyValuePair * morton,
						uint * indices,
                        uint m,
						uint d,
						uint n)
{
    uint bufL = nextPow2(n);
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(bufL, tpb);
    dim3 grid(nblk, 1, 1);
    
    computeRunHash_kernel<<< grid, block>>>(compressed,
        morton,
        indices,
        m,
        d,
        n,
        bufL);
}

void computeRunLength(uint * runLength,
							uint * runHeads,
							uint nRuns,
							uint nPrimitives,
							uint bufLength)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(bufLength, tpb);
    dim3 grid(nblk, 1, 1);
    
    computeRunLength_kernel<<< grid, block>>>(runLength,
        runHeads,
        nRuns,
        nPrimitives,
        bufLength);
}

void computeClusterAabbs(Aabb * clusterAabbs,
            Aabb * primitiveAabbs,
            uint * runHeads,
            uint * runLength,
            uint numRuns)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numRuns, tpb);
    dim3 grid(nblk, 1, 1);
    computeClusterAabbs_kernel<<< grid, block>>>(clusterAabbs,
                primitiveAabbs,
                runHeads,
                runLength,
                numRuns);
}

}
