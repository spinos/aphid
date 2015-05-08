#include "sahbvh_implement.h"
#include <bvh_math.cu>
#include "sah_math.cu"
#include <CudaBase.h>

#define ASSIGNE_EMISSIONID_NTHREAD 512
#define ASSIGNE_EMISSIONID_NTHREAD_M1 511
#define ASSIGNE_EMISSIONID_NTHREAD_LOG2 9
#define COMPUTE_BINS_NTHREAD 128
#define COMPUTE_BINS_NTHREAD_M1 127
#define COMPUTE_BINS_NTHREAD_LOG2 7

__global__ void numEmissionBlocks_kernel(uint * totalBlocks,
        EmissionBlock * emissionIds,
        EmissionEvent * inEmissions,
        int2 * rootRanges,
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
            nBlocks++;
        }
    }
    totalBlocks[0] = nBlocks;
}

__global__ void computeBins_kernel(SplitBin * splitBins,
                        Aabb * rootAabbs,
                        EmissionBlock * emissionIds,
                        Aabb * clusterAabbs,
                        uint numBins,
                        uint numClusters)
{      
    __shared__ int sSide[SAH_MAX_NUM_BINS * COMPUTE_BINS_NTHREAD];
    
    const uint iEmission = emissionIds[blockIdx.x].emission_id;
    const uint primitiveBegin = emissionIds[blockIdx.x].primitive_offset;
    
    uint ind = primitiveBegin + threadIdx.x;
	if(ind >= numClusters) return;
	
	int * sideVertical = &sSide[SAH_MAX_NUM_BINS * threadIdx.x];
	int * sideHorizontal = &sSide[threadIdx.x];
    
    Aabb rootBox = rootAabbs[iEmission];
    float * boxLow = &rootBox.low.x;
    Aabb clusterBox = clusterAabbs[ind];
    float3 center = centroidOfAabb(clusterBox);
    float * p = &center.x;
    
    const float g = longestSideOfAabb(rootBox) * .003f;
    
    computeSplitSide(sideVertical,
                        0,
                        &rootBox,
                        numBins,
                        p,
                        boxLow);
      
    __syncthreads();
    
    if(threadIdx.x < numBins)
    updateBins(splitBins,
                iEmission,
                primitiveBegin,
               clusterAabbs,
               sideHorizontal,
               rootBox.low,
               g,
               0,
               COMPUTE_BINS_NTHREAD,
               numBins,
               numClusters);
    
    computeSplitSide(sideVertical,
                        1,
                        &rootBox,
                        numBins,
                        p,
                        boxLow);
    
    __syncthreads();
    
     if(threadIdx.x < numBins)
     updateBins(splitBins,
                iEmission,
                primitiveBegin,
               clusterAabbs,
               sideHorizontal,
               rootBox.low,
               g,
               1,
               COMPUTE_BINS_NTHREAD,
               numBins,
               numClusters);

    computeSplitSide(sideVertical,
                        2,
                        &rootBox,
                        numBins,
                        p,
                        boxLow);
    
    __syncthreads();
    
     if(threadIdx.x < numBins)
     updateBins(splitBins,
                iEmission,
                primitiveBegin,
               clusterAabbs,
               sideHorizontal,
               rootBox.low,
               g,
               2,
               COMPUTE_BINS_NTHREAD,
               numBins,
               numClusters);

}

__global__ void resetBins_kernel(SplitBin * splitBins, 
                        EmissionEvent * inEmissions,
                        uint numBins)
{
    if(threadIdx.x >= numBins * 3) return;
    
    uint iEmission = blockIdx.x;
    const uint firstBin = iEmission * numBins * 3;
    
    resetSplitBin(splitBins[firstBin + threadIdx.x]);
}

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
            splitIds[primitiveRangeBegin + j].emissionId = iEmission;
    }
}

__global__ void countTreeBits_kernel(uint * nbits, 
                            KeyValuePair * morton,
                            uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
    
    nbits[ind] = 32 - __clz(morton[ind].key);
}

__global__ void writeSortedHash_kernel(KeyValuePair * dst,
							KeyValuePair * src,
							uint * indices,
							uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	dst[ind] = src[indices[ind]];
}

__global__ void computeClusterAabbs_kernel(Aabb * clusterAabbs,
            Aabb * primitiveAabbs,
            uint * compressedIndices,
            KeyValuePair * sorted,
            // uint * offset,
            uint * runLength,
            uint nRuns)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= nRuns) return;
    
    const uint sortedInd = sorted[ind].value;
	// const uint start = offset[ind];
	const uint first = compressedIndices[sortedInd];
	const uint l = runLength[ind];
    
    Aabb box;
    resetAabb(box);
    uint i = 0;
	for(;i<l;i++) 
        expandAabb(box, primitiveAabbs[first + i]);
	
    clusterAabbs[ind] = box;
}

__global__ void decompressIndices_kernel(uint * decompressedIndices,
                    uint * compressedIndices,
					KeyValuePair * sorted,
					uint * offset,
					uint * runLength,
					uint nRuns)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= nRuns) return;
	
	const uint sortedInd = sorted[ind].value;
	const uint start = offset[ind];
	const uint first = compressedIndices[sortedInd];
	const uint l = runLength[ind];
	
	uint i = 0;
	for(;i<l;i++)
	    decompressedIndices[start + i] = first + i;
}

__global__ void copyHash_kernel(KeyValuePair * dst,
					KeyValuePair * src,
					uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	dst[ind] = src[ind];
}

__global__ void compressRunHead_kernel(uint * compressed, 
							uint * runHeads,
							uint * indices,
							uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	if(runHeads[ind]) compressed[indices[ind]] = ind;
}

__global__ void computeRunLength_kernel(uint * runLength,
							uint * runHeads,
							KeyValuePair * indices,
							uint nRuns,
							uint nPrimitives,
							uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
    
	if(ind >= nRuns) {
	    runLength[ind] = 0;
	    return;
	}
	
	uint sortedInd = indices[ind].value;
	
	if(sortedInd >= nRuns-1) 
	    runLength[ind] = nPrimitives 
	                    - runHeads[sortedInd];
	else
	    runLength[ind] = runHeads[sortedInd+1] 
	                    - runHeads[sortedInd];
}

__global__ void computeRunHash_kernel(KeyValuePair * compressed, 
						KeyValuePair * morton,
						uint * indices,
                        uint m,
						uint d,
						uint maxElem,
						uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	compressed[ind].value = ind;
	
	if(ind >= maxElem)
	    compressed[ind].key = 1<<(m*3);
	else 
	    compressed[ind].key = (morton[indices[ind]].key) >> d;
}

__global__ void computeRunHead_kernel(uint * blockHeads, 
							KeyValuePair * mortonCode,
							uint d,
							uint maxElem,
							uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	if(ind >= maxElem) {
	    blockHeads[ind] = 0;
	    return;
	}
	
	if(ind < 1) {
	    blockHeads[ind] = 1;
	    return;
	}

	uint clft = mortonCode[ind - 1].key;
	uint crgt = mortonCode[ind].key;
	
	if(clft>>d == crgt>>d) blockHeads[ind] = 0;
	else blockHeads[ind] = 1;
}

extern "C" {
void sahbvh_computeRunHead(uint * blockHeads, 
							KeyValuePair * mortonCode,
							uint d,
							uint n,
							uint bufLength)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(bufLength, tpb);
    dim3 grid(nblk, 1, 1);
    
    computeRunHead_kernel<<< grid, block>>>(blockHeads,
        mortonCode,
        d,
        n,
        bufLength);
}

void sahbvh_computeRunHash(KeyValuePair * compressed, 
						KeyValuePair * morton,
						uint * indices,
                        uint m,
						uint d,
						uint n,
						uint bufLength)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(bufLength, tpb);
    dim3 grid(nblk, 1, 1);
    
    computeRunHash_kernel<<< grid, block>>>(compressed,
        morton,
        indices,
        m,
        d,
        n,
        bufLength);
}

void sahbvh_computeRunLength(uint * runLength,
							uint * runHeads,
							KeyValuePair * indices,
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
        indices,
        nRuns,
        nPrimitives,
        bufLength);
}

void sahbvh_compressRunHead(uint * compressed, 
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

void sahbvh_copyHash(KeyValuePair * dst,
					KeyValuePair * src,
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

void sahbvh_decompressIndices(uint * decompressedIndices,
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

void sahbvh_writeSortedHash(KeyValuePair * dst,
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

void sahbvh_countTreeBits(uint * nbits, 
                            KeyValuePair * morton,
                            uint n)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(n, tpb);
    dim3 grid(nblk, 1, 1);
    
    countTreeBits_kernel<<< grid, block>>>(nbits, 
                            morton,
                            n);
}

void sahbvh_computeClusterAabbs(Aabb * clusterAabbs,
            Aabb * primitiveAabbs,
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
    
    computeClusterAabbs_kernel<<< grid, block>>>(clusterAabbs, 
                            primitiveAabbs,
                            compressedIndices,
                            sorted,
                            // offset,
                            runLength,
                            n);
}

void sahbvh_numEmissionBlocks(uint * totalBinningBlocks,
        EmissionBlock * emissionIds,
        EmissionEvent * inEmissions,
        int2 * rootRanges,
        uint numEmissions)
{
    const int tpb = 1;
    dim3 block(tpb, 1, 1);
    const int nblk = 1;
    dim3 grid(nblk, 1, 1);
// one thread for all emissions
// split emission primitive into blocks
// assign emission id and block offset 
// sum up number of blocks needed
    numEmissionBlocks_kernel<<< grid, block>>>(totalBinningBlocks,
        emissionIds,
        inEmissions,
        rootRanges,
        numEmissions);

}

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

void sahbvh_resetBins(SplitBin * splitBins, 
                        EmissionEvent * inEmissions,
                        Aabb * rootAabbs,
                        uint numBins,
                        uint numEmissions)
{
    const int tpb = 128;
    dim3 block(tpb, 1, 1);
    const int nblk = numEmissions;
    dim3 grid(nblk, 1, 1);
// one block for each emission
// reset 3 * n bins
    resetBins_kernel<<< grid, block>>>(splitBins, 
        inEmissions,
        numBins);
}

void sahbvh_computeBins(SplitBin * splitBins,
                        Aabb * rootAabbs,
                        EmissionBlock * emissionIds,
                        Aabb * clusterAabbs,
                        uint numBins,
                        uint numClusters,
                        uint numBinningBlocks)
{
    const int tpb = COMPUTE_BINS_NTHREAD;
    dim3 block(tpb, 1, 1);
    const int nblk = numBinningBlocks;
    dim3 grid(nblk, 1, 1);
// one thread for each cluster/primitive
// find bins according to splitId
// atomic update bin contents

    computeBins_kernel<<< grid, block>>>(splitBins,
                        rootAabbs,
                        emissionIds,
                        clusterAabbs,
                        numBins,
                        numClusters);
}

void sahbvh_emitSahSplit(EmissionEvent * outEmissions,
	    EmissionEvent * inEmissions,
	    int2 * rootRanges,
	    Aabb * rootAabbs,
	    KeyValuePair * clusterMorton,
        Aabb * clusterAabbs,
        SplitBin * splitBins,
        EmissionBlock * emissionIds,
        SplitId * splitIds,
        uint * totalBinningBlocks,
	    uint numClusters,
        uint numBins,
	    uint numEmissions)
{
    sahbvh_assignEmissionId(splitIds,
                            inEmissions,
                            rootRanges,
                            numEmissions);
    
    sahbvh_numEmissionBlocks(totalBinningBlocks,
                        emissionIds,
                        inEmissions,
                        rootRanges,
                        numEmissions);
    
    uint numBinningBlocks = 0;
    cudaMemcpy(&numBinningBlocks, totalBinningBlocks, 4, cudaMemcpyDeviceToHost); 
    if(numBinningBlocks < 1)
        CudaBase::CheckCudaError("sah calc n binning blks");
    
    sahbvh_resetBins(splitBins, 
                        inEmissions,
                        rootAabbs,
                        numBins, 
                        numEmissions);
    
    sahbvh_computeBins(splitBins, 
                        rootAabbs,
                        emissionIds,
                        clusterAabbs, 
                        numBins,
                        numClusters,
                        numBinningBlocks);
}

}
