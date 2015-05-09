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
    const int iBin = e.bin_id;
    SplitBin & split = splitBins[iBin + iEmission * numBins * 3];
    const uint leftCount = split.leftCount;
    
    Aabb rootBox = internalNodeAabbs[iRoot];
    const float g = longestSideOfAabb(rootBox) * .003f;
    
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
    spawn->n_split = 0;
    
    spawn++;
    spawn->root_id = rightChild;
    spawn->node_offset = 0;
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
    else inEmissions[iEmission].n_split = 0;
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
	int dimension = iBin / numBins;
	float plane = splitBins[iBin + iEmission * numBins * 3].plane;
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
                                        rootArea,
                                        g);
        splitBins[iEmission * numBins * 3 
              + threadIdx.x].cost = sCost[threadIdx.x];
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
                        uint numBins,
                        uint numBinningBlocks)
{
    if(emissionIds[blockIdx.x].is_spilled < 1) return;
	if(threadIdx.x >= numBins * 3) return;
	
	const uint iEmission = emissionIds[blockIdx.x].emission_id;
    
	int isHead = 0;
	if(blockIdx.x < 1) isHead = 1;
	else if(emissionIds[blockIdx.x - 1].emission_id != iEmission) isHead = 1;
	
	if(isHead < 1) return;
	
	int i = blockIdx.x;
	for(;i<numBinningBlocks; i++) {
	    if(emissionIds[i].emission_id != iEmission) break;
	    
	    updateSplitBin(splitBins[iEmission * numBins * 3 + threadIdx.x],
	        spilledBins[emissionIds[i].bin_offset * numBins * 3 + threadIdx.x]);
	}
	
}

__global__ void computeBins_kernel(SplitBin * splitBins,
                        SplitBin * spilledBins,
                        EmissionEvent * inEmissions,
                        Aabb * rootAabbs,
                        EmissionBlock * emissionIds,
                        KeyValuePair * clusterIndirection,
                        Aabb * clusterAabbs,
                        uint numBins)
{      
    __shared__ int sSide[SAH_MAX_NUM_BINS * COMPUTE_BINS_NTHREAD];
    
    const uint iEmission = emissionIds[blockIdx.x].emission_id;
    const int iSpill = emissionIds[blockIdx.x].bin_offset;
    const uint primitiveBegin = emissionIds[blockIdx.x].primitive_offset;
    const uint iRoot = inEmissions[iEmission].root_id;
    const uint maxNumInBlock = emissionIds[blockIdx.x+1].primitive_offset;
    
    SplitBin * target = &splitBins[iEmission * numBins * 3];
    
    if(emissionIds[blockIdx.x].is_spilled) 
        target = &spilledBins[iSpill * numBins * 3];
    
    uint ind = primitiveBegin + threadIdx.x;
	
	int * sideVertical = &sSide[SAH_MAX_NUM_BINS * threadIdx.x];
	int * sideHorizontal = &sSide[threadIdx.x];
    
    Aabb rootBox = rootAabbs[iRoot];
    float * boxLow = &rootBox.low.x;
    Aabb clusterBox = clusterAabbs[clusterIndirection[ind].value];
    float3 center = centroidOfAabb(clusterBox);
    float * p = &center.x;

    if(ind < maxNumInBlock)
	    computeSplitSide(sideVertical,
                        0,
                        &rootBox,
                        numBins,
                        p,
                        boxLow);
      
    __syncthreads();
    
    if(threadIdx.x < numBins)
    updateBins(target,
                primitiveBegin,
                clusterIndirection,
               clusterAabbs,
               sideHorizontal,
               0,
               COMPUTE_BINS_NTHREAD,
               numBins,
               maxNumInBlock);
    
    __syncthreads();

    if(ind < maxNumInBlock)
        computeSplitSide(sideVertical,
                        1,
                        &rootBox,
                        numBins,
                        p,
                        boxLow);
    
    __syncthreads();
    
     if(threadIdx.x < numBins)
     updateBins(target,
                primitiveBegin,
                clusterIndirection,
               clusterAabbs,
               sideHorizontal,
               1,
               COMPUTE_BINS_NTHREAD,
               numBins,
               maxNumInBlock);
     
     __syncthreads();

     if(ind < maxNumInBlock)
        computeSplitSide(sideVertical,
                        2,
                        &rootBox,
                        numBins,
                        p,
                        boxLow);
    
    __syncthreads();
    
     if(threadIdx.x < numBins)
     updateBins(target,
                primitiveBegin,
                clusterIndirection,
               clusterAabbs,
               sideHorizontal,
               2,
               COMPUTE_BINS_NTHREAD,
               numBins,
               maxNumInBlock);
     __syncthreads();
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

void sahbvh_gatherSpillBins(SplitBin * splitBins,
                        SplitBin * spilledBins,
                        EmissionBlock * emissionIds,
                        uint numBins,
                        uint numBinningBlocks)
{
    const int tpb = 128;
    dim3 block(tpb, 1, 1);
    const int nblk = numBinningBlocks;
    dim3 grid(nblk, 1, 1);
// one block per binning block
// if it is spilled and head to event
// go forward to add up all spilled bins and set to event bin
    gatherSpillBins_kernel<<< grid, block>>>(splitBins,
                        spilledBins,
                        emissionIds,
                        numBins,
                        numBinningBlocks);
}

void sahbvh_computeBins(SplitBin * splitBins,
                        SplitBin * spilledBins,
                        EmissionEvent * inEmissions,
                        Aabb * rootAabbs,
                        EmissionBlock * emissionIds,
                        KeyValuePair * clusterIndirection,
                        Aabb * clusterAabbs,
                        uint numBins,
                        uint numBinningBlocks,
                        uint numSpilledBinBlocks)
{
    const int tpb = COMPUTE_BINS_NTHREAD;
    dim3 block(tpb, 1, 1);
    const int nblk = numBinningBlocks;
    dim3 grid(nblk, 1, 1);
// one thread for each cluster/primitive
// find bins according to splitId
// atomic update bin contents

    computeBins_kernel<<< grid, block>>>(splitBins,
                        spilledBins,
                        inEmissions,
                        rootAabbs,
                        emissionIds,
                        clusterIndirection,
                        clusterAabbs,
                        numBins);

    if(numSpilledBinBlocks > 0) {
        sahbvh_gatherSpillBins(splitBins,
                        spilledBins,
                        emissionIds,
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
        uint numSpilledBinBlocks,
        SplitBin * spilledBins,
        uint numBinningBlocks,
        uint * totalNodeCount,
	    uint numClusters,
        uint numBins,
	    uint numEmissions,
	    uint currentNumNodes)
{
    sahbvh_resetBins(splitBins,
                        numEmissions * numBins * 3);
    
    if(numSpilledBinBlocks > 0)
        sahbvh_resetBins(spilledBins,
                        numSpilledBinBlocks * numBins * 3);
    
    sahbvh_computeBins(splitBins,
                        spilledBins,
                        inEmissions,
                        rootAabbs,
                        emissionIds,
                        clusterIndirection,
                        clusterAabbs, 
                        numBins,
                        numBinningBlocks,
                        numSpilledBinBlocks);
         
    sahbvh_bestSplit(splitBins,
                    inEmissions,
                    rootAabbs,
                    numBins,
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
