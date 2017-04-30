#include "binning_implement.h"
#include "bvh_math.cuh"
#include "sah_math.cu"
#include <CudaBase.h>
#define ASSIGNE_EMISSIONID_NTHREAD 512
#define ASSIGNE_EMISSIONID_NTHREAD_M1 511
#define ASSIGNE_EMISSIONID_NTHREAD_LOG2 9

__global__ void assignEmissionId_kernel(SplitId * splitIds,
        EmissionEvent * inEmissions,
        int2 * rootRanges,
        uint nEmissions)
{
    uint iEmission = blockIdx.x;
    if(iEmission >= nEmissions) return;
    
    uint iRoot = inEmissions[iEmission].root_id;
    const int primitiveRangeBegin = rootRanges[iRoot].x;
    const int primitiveRangeEnd = rootRanges[iRoot].y;
    int numPrimitivesInRange = primitiveRangeEnd - primitiveRangeBegin + 1;
    if(numPrimitivesInRange < 0) return; // invalid range

    int npt = numPrimitivesInRange>>ASSIGNE_EMISSIONID_NTHREAD_LOG2;
    if(numPrimitivesInRange & ASSIGNE_EMISSIONID_NTHREAD_M1) npt++;
    
    int i, j;
    for(i=0; i<npt; i++) {
        j = threadIdx.x * npt + i;
        if(j < numPrimitivesInRange)
            splitIds[primitiveRangeBegin + j].emission_id = iEmission;
    }
}

__global__ void numEmissionBlocks_kernel(uint * totalBlocks,
        uint * totalSpilledBinningBlocks,
        EmissionBlock * emissionIds,
        EmissionEvent * inEmissions,
        int2 * rootRanges,
        uint numClusters,
        uint numEmissions)
{
    uint iRoot;
    int primitiveRangeBegin, primitiveRangeEnd;
    int numPrimitivesInRange, nb;
    uint nBlocks = 0;
    uint i, j;
    for(i=0;i<numEmissions;i++) {
        iRoot = inEmissions[i].root_id;
        primitiveRangeBegin = rootRanges[iRoot].x;
        primitiveRangeEnd = rootRanges[iRoot].y;
        numPrimitivesInRange = primitiveRangeEnd - primitiveRangeBegin + 1;
        
        nb = numPrimitivesInRange>>COMPUTE_BINS_NTHREAD_LOG2;
        if(numPrimitivesInRange & COMPUTE_BINS_NTHREAD_M1) nb++;
        
        for(j=0; j<nb; j++) {
            emissionIds[nBlocks].emission_id = i;
            emissionIds[nBlocks].primitive_offset = primitiveRangeBegin + j * COMPUTE_BINS_NTHREAD;
            emissionIds[nBlocks].is_spilled = (numPrimitivesInRange > COMPUTE_BINS_NTHREAD);
            emissionIds[nBlocks].bin_offset = 0;
            nBlocks++;
        }
    }
    emissionIds[nBlocks].emission_id = numEmissions - 1;
    emissionIds[nBlocks].primitive_offset = numClusters;
    emissionIds[nBlocks].is_spilled = 0;
    emissionIds[nBlocks].bin_offset = 0;
    totalBlocks[0] = nBlocks;
    
    for(i=1;i <=nBlocks;i++)
        emissionIds[i].bin_offset +=  emissionIds[i-1].bin_offset + emissionIds[i-1].is_spilled;
    
    totalSpilledBinningBlocks[0] = emissionIds[nBlocks].bin_offset;
}

extern "C" {

void sahbvh_assignEmissionId(SplitId * splitIds,
        EmissionEvent * inEmissions,
        int2 * rootRanges,
        uint numEmissions)
{
    const int tpb = ASSIGNE_EMISSIONID_NTHREAD;
    dim3 block(tpb, 1, 1);
    const int nblk = numEmissions;
    dim3 grid(nblk, 1, 1);
// one block for each emission
// assign emissionIds to each primitive in range
    assignEmissionId_kernel<<< grid, block>>>(splitIds,
        inEmissions,
        rootRanges,
        numEmissions);
}

void sahbvh_numEmissionBlocks(uint * totalBinningBlocks,
        uint * totalSpilledBinningBlocks,
        EmissionBlock * emissionIds,
        EmissionEvent * inEmissions,
        int2 * rootRanges,
        uint numClusters,
        uint numEmissions)
{
    const int tpb = 1;
    dim3 block(tpb, 1, 1);
    const int nblk = 1;
    dim3 grid(nblk, 1, 1);
// one thread for all emissions
// split emission primitive into blocks
// assign emission id and block offset 
// sum up number of blocks needed, add last to hold total num primitives
// sum up spilled blocks
    numEmissionBlocks_kernel<<< grid, block>>>(totalBinningBlocks,
        totalSpilledBinningBlocks,
        emissionIds,
        inEmissions,
        rootRanges,
        numClusters,
        numEmissions);

}

void sahbvh_getNumBinningBlocks(uint * numBinningBlocks,
                        uint * numSpilledBlocks,
                        uint * totalBinningBlocks,
                        uint * totalSpilledBinningBlocks,
                        SplitId * splitIds,
                        EmissionBlock * emissionIds,
                        EmissionEvent * inEmissions,
                        int2 * rootRanges,
                        uint numClusters,
                        uint numEmissions)
{
    sahbvh_assignEmissionId(splitIds,
                            inEmissions,
                            rootRanges,
                            numEmissions);
    
    sahbvh_numEmissionBlocks(totalBinningBlocks,
                        totalSpilledBinningBlocks,
                        emissionIds,
                        inEmissions,
                        rootRanges,
                        numClusters,
                        numEmissions);
    
    *numBinningBlocks = 0;
    cudaMemcpy(numBinningBlocks, totalBinningBlocks, 4, cudaMemcpyDeviceToHost); 
    if(*numBinningBlocks < 1)
        CudaBase::CheckCudaError("sah calc n binning blks");
    
    *numSpilledBlocks = 0;
    cudaMemcpy(numSpilledBlocks, totalSpilledBinningBlocks, 4, cudaMemcpyDeviceToHost); 
    
}

}

