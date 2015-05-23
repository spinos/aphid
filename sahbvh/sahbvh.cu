#include "sahbvh_implement.h"
#include <bvh_math.cu>
#include "sah_math.cu"
#include <CudaBase.h>

__global__ void spawnNode_kernel(int2 * internalNodeChildIndices,
                    Aabb * internalNodeAabbs,
                    EmissionEvent * outEmissions,
                    EmissionEvent * inEmissions,
                    SplitBin * splitBins,
                    uint numBins,
                    uint currentNumNodes,
                    uint * totalNodeCountAft)
{
    const uint iEmission = blockIdx.x;
    EmissionEvent & e = inEmissions[iEmission];
    const uint iRoot = e.root_id;
    SplitBin & split = splitBins[iEmission];
    const uint leftCount = split.leftCount;
    
    Aabb rootBox = internalNodeAabbs[iRoot];
    // const float g = longestSideOfAabb(rootBox) * .003f;
    
    int2 & rootInd = internalNodeChildIndices[iRoot];
    
    const int primitiveBegin = rootInd.x;
    const int primitiveEnd = rootInd.y;
    
    int leftChild = currentNumNodes + e.node_offset;
    int rightChild = leftChild + 1;
    
    internalNodeChildIndices[leftChild].x = primitiveBegin;
    internalNodeChildIndices[leftChild].y = primitiveBegin + leftCount - 1;
    
    internalNodeChildIndices[rightChild].x = primitiveBegin + leftCount;
    internalNodeChildIndices[rightChild].y = primitiveEnd;
    
    internalNodeAabbs[leftChild] = split.leftBox;
    internalNodeAabbs[rightChild] = split.rightBox;
    
    //binAabbToAabb(internalNodeAabbs[leftChild], split.leftBox, 
      //  rootBox.low, g);
    //binAabbToAabb(internalNodeAabbs[rightChild], split.rightBox, 
      //  rootBox.low, g);
  
// mark as internal
    rootInd.x = (leftChild | 0x80000000);
    rootInd.y = (rightChild| 0x80000000);
    
    EmissionEvent * spawn = &outEmissions[e.node_offset];
    spawn->root_id = leftChild;
    spawn->node_offset = 0;
    spawn->bin_id = -1;
    spawn->n_split = 0;
    
    spawn++;
    spawn->root_id = rightChild;
    spawn->node_offset = 0;
    spawn->bin_id = -1;
    spawn->n_split = 0;
}

__global__ void scanNodeOffset_kernel(uint * nodeCount,
                            EmissionEvent * inEmissions,
                            uint numEmissions)
{
    uint i;
    for(i=1;i<numEmissions;i++) {
        inEmissions[i].node_offset = inEmissions[i-1].node_offset
                                    + inEmissions[i-1].n_split * 2;
    }
    nodeCount[0] += inEmissions[numEmissions-1].node_offset 
                    + inEmissions[numEmissions-1].n_split * 2;
}

__global__ void splitIndirection_kernel(KeyValuePair * primitiveInd,
                            SplitId * splitIds,
                            EmissionEvent * inEmissions,
                            int2 * rootRanges)
{
    const uint iEmission = blockIdx.x;
    if(inEmissions[iEmission].bin_id < 0) return;
    const uint iRoot = inEmissions[iEmission].root_id;
    const int rootEnd = rootRanges[iRoot].y;
    int rangeBegin = rootRanges[iRoot].x;
    int rangeEnd = rootEnd;
    KeyValuePair intermediate;
    for(;;) {
        while(splitIds[rangeBegin].split_side<1) rangeBegin++;
        while(splitIds[rangeEnd].split_side>0) rangeEnd--;
            
        if(rangeBegin >= rangeEnd) break;
        
        intermediate = primitiveInd[rangeBegin];
        primitiveInd[rangeBegin++] = primitiveInd[rangeEnd];
        primitiveInd[rangeEnd--] = intermediate;
    }
    if(rangeBegin < rootEnd) inEmissions[iEmission].n_split = 1;
}

__global__ void computeSplitSide_kernel(SplitId * splitIds,
                            EmissionEvent * inEmissions,
                            KeyValuePair * clusterInd,
                            Aabb * clusterAabbs,
                            SplitBin * splitBins,
                            uint numBins,
                            uint numClusters)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= numClusters) return;
	
	const uint iEmission = splitIds[ind].emission_id;
	const int iBin = inEmissions[iEmission].bin_id;
	if(iBin<0) return;
	int dimension = iBin / numBins;
	float plane = splitBins[iEmission].plane;
    Aabb clusterBox = clusterAabbs[clusterInd[ind].value];
    
    float3 center = centroidOfAabb(clusterBox);
    if(float3_component(center, dimension) <= plane) splitIds[ind].split_side = 0;
    else splitIds[ind].split_side = 1;
}

__global__ void bestSplit_kernel(SplitBin * splitBins, 
                        EmissionEvent * inEmissions,
                        Aabb * rootAabbs,
                        uint numBins)
{
    __shared__ float sCost[SAH_MAX_NUM_BINS * 16];
    
    const uint iEmission = blockIdx.x;
    const uint iRoot = inEmissions[iEmission].root_id;
    Aabb rootBox = rootAabbs[iRoot];
    const float rootArea = areaOfAabb(&rootAabbs[iRoot]);
    const float g = longestSideOfAabb(rootBox) * .003f;
    
    if(threadIdx.x < numBins * 3) {
        sCost[threadIdx.x] = costOfSplit(&splitBins[iEmission * numBins * 3 
                                                    + threadIdx.x],
                                        rootArea);
        
        //splitBins[iEmission * numBins * 3 
          //    + threadIdx.x].cost = sCost[threadIdx.x];
    }
    __syncthreads();
    
    int i;
    int splitI;
    float minCost;
    if(threadIdx.x < 1) {
        minCost = sCost[0];
        splitI = 0;
        for(i=1; i < numBins * 3; i++) {
            if(minCost > sCost[i]) {
                minCost = sCost[i];
                splitI = i;
            }
        }
        
        inEmissions[iEmission].bin_id = splitI;
        splitBins[splitI
                  + iEmission * numBins * 3].plane = splitPlaneOfBin(&rootBox,
                                                numBins,
                                                splitI);
        
    }
}

__global__ void gatherSpillBins_kernel(SplitBin * splitBins,
                        SplitBin * spilledBins,
                        EmissionBlock * emissionIds,
                        EmissionEvent * inEmissions,
                        Aabb * rootAabbs,
                        uint numBins,
                        uint numBinningBlocks)
{
    __shared__ SplitBin sGathered[SAH_MAX_NUM_BINS * 3];
    __shared__ float sCost[SAH_MAX_NUM_BINS * 3];
    
    if(emissionIds[blockIdx.x].is_spilled < 1) return;
    
    const uint numB3 = numBins * 3;
    const uint iEmission = emissionIds[blockIdx.x].emission_id;
    const uint iRoot = inEmissions[iEmission].root_id;
    const int spilled = emissionIds[blockIdx.x].is_spilled;
    Aabb rootBox = rootAabbs[iRoot];
    const float rootArea = areaOfAabb(&rootBox);
    
	int isHead = 0;
	if(spilled) {
	    if(blockIdx.x < 1) isHead = 1;
	    else if(emissionIds[blockIdx.x - 1].emission_id != iEmission) isHead = 1;
	    if(isHead < 1) return;
	}
	
	int i;
	
	if(threadIdx.x < numB3) {
	    if(spilled) {
	        resetSplitBin(sGathered[threadIdx.x]);
	        
	        for(i=blockIdx.x;i<numBinningBlocks; i++) {
	            if(emissionIds[i].emission_id != iEmission) break;
            
	            updateSplitBin(sGathered[threadIdx.x],
	                            spilledBins[emissionIds[i].bin_offset * numB3 
	                                        + threadIdx.x]);
            }
        }
        else 
            sGathered[threadIdx.x] = spilledBins[blockIdx.x * numB3 
                                                + threadIdx.x];
	
    }
	__syncthreads();
	
	if(threadIdx.x < numB3)
	    sCost[threadIdx.x] = costOfSplit(&sGathered[threadIdx.x],
                                        rootArea);
	__syncthreads();
    
	if(threadIdx.x < 1) {
	    int bestBin = -1;
	    float lowestCost = 1e20f;
        for(i=0; i< numB3; i++) {
            if(lowestCost > sCost[i]) {
                lowestCost = sCost[i];
                bestBin = i;
            }
        }
        
        if(bestBin < 0) return;
        
        sGathered[bestBin].plane = splitPlaneOfBin(&rootBox,
                                                numBins,
                                                bestBin);
        
        splitBins[iEmission] = sGathered[bestBin];
        inEmissions[iEmission].bin_id = bestBin;
	}
}

__global__ void computeBinsSpread_kernel(SplitBin * spilledBins,
                        EmissionEvent * inEmissions,
                        int2 * rootRanges,
                        Aabb * rootAabbs,
                        EmissionBlock * emissionIds,
                        KeyValuePair * clusterIndirection,
                        Aabb * clusterAabbs,
                        uint numBins)
{      
    __shared__ int sSide[SAH_MAX_NUM_BINS * COMPUTE_BINS_NTHREAD];    
    __shared__ SplitBin sBin[SAH_MAX_NUM_BINS];
    
    const int binBlkI = blockIdx.x / 3;
    if(emissionIds[binBlkI].is_spilled < 1) return;
    
    const int outLoc = emissionIds[binBlkI].bin_offset;
    const uint iEmission = emissionIds[binBlkI].emission_id;
    const uint dimension = blockIdx.x - binBlkI * 3;
    
    const uint iRoot = inEmissions[iEmission].root_id;
    
    const int primitiveStartInBlock = emissionIds[binBlkI].primitive_offset;
    const int primitiveEndInBlock = emissionIds[binBlkI+1].primitive_offset;
    const int primitiveCountInBlock = primitiveEndInBlock - primitiveStartInBlock;
    
    Aabb rootBox = rootAabbs[iRoot];
    const float rootArea = areaOfAabb(&rootBox);
    
	int * sideVertical = &sSide[SAH_MAX_NUM_BINS * threadIdx.x];
	int * sideHorizontal = &sSide[threadIdx.x];
    
	float boxLow = float3_component(rootBox.low, dimension);
    Aabb clusterBox;
    float3 center;
    float p;

    if(threadIdx.x < numBins)
        resetSplitBin(sBin[threadIdx.x]);
       
    int j = primitiveStartInBlock + threadIdx.x;
        
    if(j<primitiveEndInBlock) { 
            
        clusterBox = clusterAabbs[clusterIndirection[j].value];
        center = centroidOfAabb(clusterBox);
        p = float3_component(center, dimension);
        
        computeSplitSide(sideVertical,
                    dimension,
                    &rootBox,
                    numBins,
                    p,
                    boxLow);
    }
        
    __syncthreads();
    
    if(threadIdx.x < numBins) {
            collectBins(sBin[threadIdx.x],
                clusterIndirection,
               clusterAabbs,
               sideHorizontal,
               primitiveStartInBlock,
               primitiveEndInBlock);
            
            spilledBins[outLoc * numBins * 3
                    + dimension * numBins
                    + threadIdx.x] = sBin[threadIdx.x];
    }
}

__global__ void computeBinsInBlock_kernel(SplitBin * outBins,
                        EmissionEvent * inEmissions,
                        int2 * rootRanges,
                        Aabb * rootAabbs,
                        EmissionBlock * emissionIds,
                        KeyValuePair * clusterIndirection,
                        Aabb * clusterAabbs,
                        uint numBins)
{
    __shared__ int sSide[SAH_MAX_NUM_BINS * COMPUTE_BINS_NTHREAD];    
    __shared__ SplitBin sBin[SAH_MAX_NUM_BINS];
    __shared__ float sCost[SAH_MAX_NUM_BINS];
    
    if(emissionIds[blockIdx.x].is_spilled) return;
    
    const uint iEmission = emissionIds[blockIdx.x].emission_id;
    const uint iRoot = inEmissions[iEmission].root_id;
    
    const int primitiveStartInBlock = emissionIds[blockIdx.x].primitive_offset;
    const int primitiveEndInBlock = emissionIds[blockIdx.x+1].primitive_offset;
    const int primitiveCountInBlock = primitiveEndInBlock - primitiveStartInBlock;
    
    Aabb rootBox = rootAabbs[iRoot];
    const float rootArea = areaOfAabb(&rootBox);
    
	int * sideVertical = &sSide[SAH_MAX_NUM_BINS * threadIdx.x];
	int * sideHorizontal = &sSide[threadIdx.x];
    
	float boxLow;
    Aabb clusterBox;
    float3 center;
    float p;
    SplitBin bestBin;
    resetSplitBin(bestBin);
    
    int bestI = -1;
    float lowestCost = 1e22f;    
    int i;
    const int j = primitiveStartInBlock + threadIdx.x;
        
    if(j<primitiveEndInBlock) {
        
        boxLow = float3_component(rootBox.low, 0);    
        clusterBox = clusterAabbs[clusterIndirection[j].value];
        center = centroidOfAabb(clusterBox);
        p = float3_component(center, 0);
        
        computeSplitSide(sideVertical,
                    0,
                    &rootBox,
                    numBins,
                    p,
                    boxLow);
    }
    
    __syncthreads();
    
    if(threadIdx.x < numBins) {
            resetSplitBin(sBin[threadIdx.x]);
            collectBins(sBin[threadIdx.x],
                clusterIndirection,
               clusterAabbs,
               sideHorizontal,
               primitiveStartInBlock,
               primitiveEndInBlock);
            
            sCost[threadIdx.x] = costOfSplit(&sBin[threadIdx.x],
                                        rootArea);
    }
    
    __syncthreads();
    
    if(threadIdx.x < 1) {
        for(i=0; i< numBins; i++) {
            if(lowestCost > sCost[i]) {
                lowestCost = sCost[i];
                bestI = i;
            }
        }
        
        if(bestI >= 0) {
            bestBin = sBin[bestI];
            bestBin.plane = splitPlaneOfBin(&rootBox,
                                                numBins,
                                                bestI);
            bestBin.dimension = bestI;
        }
    }
    
    if(j<primitiveEndInBlock) {
        
        boxLow = float3_component(rootBox.low, 1);    
        clusterBox = clusterAabbs[clusterIndirection[j].value];
        center = centroidOfAabb(clusterBox);
        p = float3_component(center, 1);
        
        computeSplitSide(sideVertical,
                    1,
                    &rootBox,
                    numBins,
                    p,
                    boxLow);
    }
    
    __syncthreads();
    
    if(threadIdx.x < numBins) {
            resetSplitBin(sBin[threadIdx.x]);
            collectBins(sBin[threadIdx.x],
                clusterIndirection,
               clusterAabbs,
               sideHorizontal,
               primitiveStartInBlock,
               primitiveEndInBlock);
            
            sCost[threadIdx.x] = costOfSplit(&sBin[threadIdx.x],
                                        rootArea);
    }
    
    __syncthreads();
    
    if(threadIdx.x < 1) {
        bestI = -1;
        for(i=0; i< numBins; i++) {
            if(lowestCost > sCost[i]) {
                lowestCost = sCost[i];
                bestI = i;
            }
        }
        
        if(bestI >= 0) {
            bestBin = sBin[bestI];
            bestBin.plane = splitPlaneOfBin(&rootBox,
                                                numBins,
                                                numBins + bestI);
            bestBin.dimension = numBins + bestI;
        }
    }
    
    if(j<primitiveEndInBlock) {
        
        boxLow = float3_component(rootBox.low, 2);    
        clusterBox = clusterAabbs[clusterIndirection[j].value];
        center = centroidOfAabb(clusterBox);
        p = float3_component(center, 2);
        
        computeSplitSide(sideVertical,
                    2,
                    &rootBox,
                    numBins,
                    p,
                    boxLow);
    }
    
    __syncthreads();
    
    if(threadIdx.x < numBins) {
            resetSplitBin(sBin[threadIdx.x]);
            collectBins(sBin[threadIdx.x],
                clusterIndirection,
               clusterAabbs,
               sideHorizontal,
               primitiveStartInBlock,
               primitiveEndInBlock);
            
            sCost[threadIdx.x] = costOfSplit(&sBin[threadIdx.x],
                                        rootArea);
    }
    
    __syncthreads();
    
    if(threadIdx.x < 1) {
        bestI = -1;
        for(i=0; i< numBins; i++) {
            if(lowestCost > sCost[i]) {
                lowestCost = sCost[i];
                bestI = i;
            }
        }
        
        if(bestI >= 0) {
            bestBin = sBin[bestI];
            bestBin.plane = splitPlaneOfBin(&rootBox,
                                                numBins,
                                                numBins * 2 + bestI);
            bestBin.dimension = numBins*2 + bestI;
        }
    }
    
    if(threadIdx.x < 1) {
        outBins[iEmission] = bestBin;
        inEmissions[iEmission].bin_id = bestBin.dimension;
    }
}

__global__ void resetBins_kernel(SplitBin * splitBins, 
                        uint n)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= n) return;
    
    resetSplitBin(splitBins[ind]);
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
	
    clusterAabbs[sortedInd] = box;
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
	
	if(ind>=maxElem)
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

void sahbvh_resetBins(SplitBin * splitBins, 
                        uint n)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    const int nblk = iDivUp(n, tpb);
    dim3 grid(nblk, 1, 1);

    resetBins_kernel<<< grid, block>>>(splitBins, 
        n);
}

void sahbvh_computeBins(SplitBin * splitBins,
                        SplitBin * spilledBins,
                        EmissionEvent * inEmissions,
                        int2 * rootRanges,
                        Aabb * rootAabbs,
                        EmissionBlock * emissionIds,
                        KeyValuePair * clusterIndirection,
                        Aabb * clusterAabbs,
                        uint numBins,
                        uint numBinningBlocks,
                        uint numSpilledBinBlocks,
                        uint numEmissions)
{
    const int tpb = COMPUTE_BINS_NTHREAD;
    dim3 block(tpb, 1, 1);
    const int nblk = numBinningBlocks * 3;
    dim3 grid(nblk, 1, 1);
// one block per dimension for each bin block
// if spilled write to spilled bins

    const int tpbGather = 64;
    dim3 blockGather(tpbGather, 1, 1);
    dim3 gridGather(numBinningBlocks, 1, 1);
// one block per bin block
// find the best split in 3 dimensions

    const int tpbIb = COMPUTE_BINS_NTHREAD;
    dim3 blockIb(tpbIb, 1, 1);
    dim3 gridIb(numBinningBlocks, 1, 1);
    
    if(numSpilledBinBlocks < numBinningBlocks) {
        computeBinsInBlock_kernel<<<gridIb, blockIb>>>(splitBins,
                        inEmissions,
                        rootRanges,
                        rootAabbs,
                        emissionIds,
                        clusterIndirection,
                        clusterAabbs,
                        numBins);
    }
    
    if(numSpilledBinBlocks > 0) {
        computeBinsSpread_kernel<<<grid, block>>>(spilledBins,
                        inEmissions,
                        rootRanges,
                        rootAabbs,
                        emissionIds,
                        clusterIndirection,
                        clusterAabbs,
                        numBins);
      
    
        gatherSpillBins_kernel<<<gridGather, blockGather>>>(splitBins,
                        spilledBins,
                        emissionIds,
                        inEmissions,
                        rootAabbs,
                        numBins,
                        numBinningBlocks);
    }
}

void sahbvh_bestSplit(SplitBin * splitBins, 
                        EmissionEvent * inEmissions,
                        Aabb * rootAabbs,
                        uint numBins,
                        uint numEmissions)
{
    const int tpb = 128;
    dim3 block(tpb, 1, 1);
    const int nblk = numEmissions;
    dim3 grid(nblk, 1, 1);
// one block per emission
// find the lowest split cost
    bestSplit_kernel<<< grid, block>>>(splitBins,
                                    inEmissions,
                                    rootAabbs,
                                    numBins);
}

void sahbvh_computeSplitSide(SplitId * splitIds,
                            EmissionEvent * inEmissions,
                            KeyValuePair * clusterIndirection,
                            Aabb * clusterAabbs,
                            SplitBin * splitBins,
                            uint numBins,
                            uint numClusters)
{
    const int tpb = 512;
    dim3 block(tpb, 1, 1);
    const int nblk = iDivUp(numClusters, tpb);
    dim3 grid(nblk, 1, 1);
// one thread per primitive
// decide left or right it reside to the split plane
    computeSplitSide_kernel<<< grid, block>>>(splitIds,
                                    inEmissions,
                                    clusterIndirection,
                                    clusterAabbs,
                                    splitBins,
                                    numBins,
                                    numClusters);
}

void sahbvh_splitIndirection(KeyValuePair * primitiveInd,
                            SplitId * splitIds,
                            EmissionEvent * inEmissions,
                            int2 * rootRanges,
                            uint numEmissions)
{
    const int tpb = 1;
    dim3 block(tpb, 1, 1);
    const int nblk = numEmissions;
    dim3 grid(nblk, 1, 1);
// one block per emission
// within range of root node
// swap any ind that split side 1 before side 0
    splitIndirection_kernel<<< grid, block>>>(primitiveInd,
                                            splitIds,
                                            inEmissions,
                                            rootRanges);
}

void sahbvh_scanNodeOffset(uint * nodeCount,
                            EmissionEvent * inEmissions,
                            uint numEmissions)
{
    const int tpb = 1;
    dim3 block(tpb, 1, 1);
    const int nblk = 1;
    dim3 grid(nblk, 1, 1);
// one thread prefix sum node offset
// add to nodeCount
    scanNodeOffset_kernel<<< grid, block>>>(nodeCount,
                                            inEmissions,
                                            numEmissions);
}

void sahbvh_spawnNode(int2 * internalNodeChildIndices,
                    Aabb * internalNodeAabbs,
                    EmissionEvent * outEmissions,
                    EmissionEvent * inEmissions,
                    SplitBin * splitBins,
                    uint numBins,
                    uint numEmissions,
                    uint currentNumNodes,
                    uint * totalNodeCount)
{
    const int tpb = 1;
    dim3 block(tpb, 1, 1);
    const int nblk = numEmissions;
    dim3 grid(nblk, 1, 1);
// one thread per emission
// set child node range
// connect to root
// create next level emissions
    spawnNode_kernel<<< grid, block>>>(internalNodeChildIndices,
                                    internalNodeAabbs,
                                    outEmissions,
                                    inEmissions,
                                    splitBins,
                                    numBins,
                                    currentNumNodes,
                                    totalNodeCount);
}

void sahbvh_emitSahSplit(EmissionEvent * outEmissions,
	    EmissionEvent * inEmissions,
	    int2 * rootRanges,
	    Aabb * rootAabbs,
	    KeyValuePair * clusterIndirection,
        Aabb * clusterAabbs,
        SplitBin * splitBins,
        EmissionBlock * emissionIds,
        SplitId * splitIds,
        uint numBinningBlocks,
        uint numSpilledBinBlocks,
        uint * totalNodeCount,
	    uint numClusters,
        uint numBins,
	    uint numEmissions,
	    uint currentNumNodes)
{
    sahbvh_resetBins(splitBins,
                        numEmissions);
    
    sahbvh_computeBins(splitBins,
                        &splitBins[numEmissions],
                        inEmissions,
                        rootRanges,
                        rootAabbs,
                        emissionIds,
                        clusterIndirection,
                        clusterAabbs, 
                        numBins,
                        numBinningBlocks,
                        numSpilledBinBlocks,
                        numEmissions);
              
    sahbvh_computeSplitSide(splitIds,
                            inEmissions,
                            clusterIndirection,
                            clusterAabbs,
                            splitBins,
                            numBins,
                            numClusters);
    
    sahbvh_splitIndirection(clusterIndirection,
                            splitIds,
                            inEmissions,
                            rootRanges,
                            numEmissions);
    
    sahbvh_scanNodeOffset(totalNodeCount,
                            inEmissions,
                            numEmissions);
                            
    sahbvh_spawnNode(rootRanges,
                    rootAabbs,
                    outEmissions,
                    inEmissions,
                    splitBins,
                    numBins,
                    numEmissions,
                    currentNumNodes,
                    totalNodeCount);                    
}

}
