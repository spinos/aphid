#ifndef OVERLAPPING2_CUH
#define OVERLAPPING2_CUH

/*
 *  collision packet traverse
 */

#include "bvhUtil.cuh"

template<int NumThreadsPerDim>
__device__ void findOveralppings(int * scounts,
                                         Aabb box,
                                         int m,
                                         int n,
                                         Aabb * elementBoxes,
                                         uint * elementInd,
                                         uint tid)
{
/*
 *  layout of shared counts
 *  n as num threads per dimension
 *  nn threads
 *  horizontal is idx of primitive to query
 *  vertical is idx of primitive to test
 *
 *  0  1    2    3  ... n-1
 *  n  n+1  n+2  n+3    2n-1
 *  2n 2n+2 2n+2 2n+3   3n-1
 *  ...
 *  (n-1)n ...          nn-1
 */
    scounts[tid] = 0;
    if(threadIdx.x < m && threadIdx.y < n) {
        if(isAabbOverlapping(box, elementBoxes[threadIdx.y])) {
                scounts[tid] = 1;
        }
    }
}

template<int NumThreadsPerDim>
__device__ void findOveralppingsWId(int * scounts,
                                        uint * selmIds,
                                         Aabb box,
                                         int m,
                                         int n,
                                         Aabb * elementBoxes,
                                         uint * elementInd,
                                         uint tid,
                                         uint iQuery)
{
    scounts[tid] = 0;
    uint iElement;
    if(threadIdx.x < m && threadIdx.y < n) {
        iElement = elementInd[threadIdx.y];
        if(isAabbOverlapping(box, elementBoxes[threadIdx.y])) {
            scounts[tid] = 1;
            selmIds[tid] = combineObjectElementInd(iQuery, iElement);
        }
    }
}

template<int NumThreadsPerDim>
__global__ void countPairsPacket_kernel(uint * overlappingCounts, 
                                Aabb * boxes,
                                KeyValuePair * queryIndirection,
                                int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, 
								Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices)
{
    int *sdata = SharedMemory<int>();
    
	uint queryInd = blockIdx.x;
    int2 queryRange = internalNodeChildIndices[queryInd];
    if(!isLeafNode(queryRange)) return;
    
	const uint tid = tId2();
	int querySize = queryRange.y - queryRange.x + 1;
	const int isValidBox = (tid < querySize);
/* 
 *  smem layout in ints
 *  n as max num primitives in a leaf node 16
 *  m as max stack size 64
 *
 *  0   -> 1      stackSize
 *  1   ->       visit left child
 *  2   ->       visit right child
 *  4   -> 4+m-1       stack
 *  4+m -> 4+m+n-1  query index
 *  4+m+n -> 4+m+n+6n-1  query boxes
 *  4+m+n+6n -> 4+m+n+12n-1  leaf boxes cache
 *  4+m+n+12n -> 4+m+n+12n+n-1 leaf boxes ind
 *  4+m+n+12n+n -> 4+m+n+12n+n+nn-1 overlapping counts
 *
 */		
    int & sstackSize =          sdata[0];
    int & b1 =                  sdata[1];
    int & b2 =                  sdata[2];
    int * sstack =             &sdata[4];
    uint * squeryIdx = (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE];
    Aabb * squeryBox = (Aabb *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim];
    Aabb * sboxCache = (Aabb *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim + 6 * NumThreadsPerDim];
    uint * sindCache = (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim + 12 * NumThreadsPerDim];
    int * scounts =            &sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim + 12 * NumThreadsPerDim + NumThreadsPerDim];
	
    if(isValidBox) {
	    squeryIdx[tid] = mortonCodesAndAabbIndices[queryRange.x + tid].value;
        squeryBox[tid] = boxes[squeryIdx[tid]];
    }
    
    if(tid<1) {
        sstack[0] = 0x80000000;
        sstackSize = 1;
    }
    __syncthreads();
    	
    int numBoxesInLeaf;
	int isLeaf;
    int iNode;
    int2 child;
    Aabb leftBox, rightBox;
    
    const uint boxInd = squeryIdx[threadIdx.x];
    const Aabb box = squeryBox[threadIdx.x];
    const Aabb groupBox = internalNodeAabbs[queryInd];
    
    uint outCount;
    if(threadIdx.x < querySize) outCount = overlappingCounts[boxInd];
	
	for(;;) {
		iNode = sstack[ sstackSize - 1 ];
		iNode = getIndexWithInternalNodeMarkerRemoved(iNode);
        child = internalNodeChildIndices[iNode];
        isLeaf = isLeafNode(child);
			
		if(isLeaf) {
// load leaf boxes into smem
            numBoxesInLeaf = child.y - child.x + 1;
            if(numBoxesInLeaf > NumThreadsPerDim) numBoxesInLeaf = NumThreadsPerDim;
            putLeafBoxInSmem<NumThreadsPerDim>(sboxCache,
                                            tid,
                                            numBoxesInLeaf,
                                            child,
                                            mortonCodesAndAabbIndices,
                                            leafAabbs);
            __syncthreads();
// intersect boxes in leaf using all threads
            findOveralppings<NumThreadsPerDim>(scounts,
                                    box,
                                    querySize,
                                    numBoxesInLeaf,
                                    sboxCache,
                                    sindCache,
                                    tid);
            __syncthreads();
            if(isValidBox)
                sumOverlappingCounts<NumThreadsPerDim>(outCount,
                                    scounts,
                                    numBoxesInLeaf,
                                    tid);
            
            if(tid<1) {
// take out top of stack
                sstackSize--;
            }
            __syncthreads();
            
            if(sstackSize<1) break;
		}
		else {
            if(tid<1) {
                leftBox = internalNodeAabbs[getIndexWithInternalNodeMarkerRemoved(child.x)];
                b1 = isAabbOverlapping(groupBox, leftBox);
            }
            else if(tid==1) {
                rightBox = internalNodeAabbs[getIndexWithInternalNodeMarkerRemoved(child.y)];
                b2 = isAabbOverlapping(groupBox, rightBox);
            }
            __syncthreads();
            
            if(tid<1) {
                if(b1 == 0 && b2 == 0) {
// visit no child, take out top of stack
                    sstackSize--;
                }
                else if(b1 > b2) {
// visit right child
                    sstack[ sstackSize - 1 ] = child.x; 
                }
                else if(b2 > b1) {
// visit right child
                    sstack[ sstackSize - 1 ] = child.y; 
                }
                else {
// visit both children                    
                    sstack[ sstackSize - 1 ] = child.y;
                    if(sstackSize < BVH_TRAVERSE_MAX_STACK_SIZE) { 
                            sstack[ sstackSize ] = child.x;
                            sstackSize++;
                    }
                }
            }
            __syncthreads();
            
            if(sstackSize<1) break;
		}
	}
	if(isValidBox) 
        overlappingCounts[boxInd] = outCount;
}

template<int NumThreadsPerDim>
__global__ void writePairCachePacket_kernel(uint2 * outPairs, 
                                uint * cacheWriteLocation,
                                Aabb * boxes,
                                KeyValuePair * queryIndirection,
                                uint maxBoxInd,
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, 
								Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								uint queryIdx, 
								uint treeIdx)
{
    int *sdata = SharedMemory<int>();
    
    uint queryInd = blockIdx.x;
    int2 queryRange = internalNodeChildIndices[queryInd];
    if(!isLeafNode(queryRange)) return;
    
	const uint tid = tId2();
	int querySize = queryRange.y - queryRange.x + 1;
	const int isValidBox = (tid < querySize);
/* 
 *  smem layout in ints
 *  n as max num primitives in a leaf node 16
 *  m as max stack size 64
 *
 *  0   -> 1      stackSize
 *  1   ->       visit left child
 *  2   ->       visit right child
 *  4   -> 4+m-1       stack
 *  4+m -> 4+m+n-1  query index
 *  4+m+n -> 4+m+n+6n-1  query boxes
 *  4+m+n+6n -> 4+m+n+12n-1  leaf boxes cache
 *  4+m+n+12n -> 4+m+n+12n+n-1 leaf boxes ind
 *  4+m+n+12n+n -> 4+m+n+12n+n+nn-1 overlapping counts
 *  4+m+n+12n+n+nn -> 4+m+n+12n+n+nn+nn-1 overlapping ids
 *
 */	
    int & sstackSize =          sdata[0];
    int & b1 =                  sdata[1];
    int & b2 =                  sdata[2];
    int * sstack =             &sdata[4];
    uint * squeryIdx = (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE];
    Aabb * squeryBox = (Aabb *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim];
    Aabb * sboxCache = (Aabb *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim + 6 * NumThreadsPerDim];
    uint * sindCache = (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim + 12 * NumThreadsPerDim];
    int * scounts =            &sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim + 12 * NumThreadsPerDim + NumThreadsPerDim];
	uint * selmIds =   (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim + 12 * NumThreadsPerDim + NumThreadsPerDim + NumThreadsPerDim * NumThreadsPerDim]; 
	
	if(isValidBox) {
	    squeryIdx[tid] = mortonCodesAndAabbIndices[queryRange.x + tid].value;
        squeryBox[tid] = boxes[squeryIdx[tid]];
    }

	if(tid<1) {
        sstack[0] = 0x80000000;
        sstackSize = 1;
    }
    __syncthreads();
	
    int numBoxesInLeaf;
	int isLeaf;
    int iNode;
    int2 child;
    Aabb leftBox, rightBox;
    
    const uint boxInd = squeryIdx[threadIdx.x];
    const Aabb box = squeryBox[threadIdx.x];
    const Aabb groupBox = internalNodeAabbs[queryInd];
    uint writeLoc;
    if(threadIdx.x < querySize) writeLoc = cacheWriteLocation[boxInd];
	
	for(;;) {
		iNode = sstack[ sstackSize - 1 ];
		iNode = getIndexWithInternalNodeMarkerRemoved(iNode);
        child = internalNodeChildIndices[iNode];
        isLeaf = isLeafNode(child);
        
		if(isLeaf) {
// load leaf boxes into smem
            numBoxesInLeaf = child.y - child.x + 1;
            if(numBoxesInLeaf > NumThreadsPerDim) numBoxesInLeaf = NumThreadsPerDim;
            putLeafBoxAndIndInSmem<NumThreadsPerDim>(sboxCache,
                                            sindCache,
                                            tid,
                                            numBoxesInLeaf,
                                            child,
                                            mortonCodesAndAabbIndices,
                                            leafAabbs);
            __syncthreads();
// intersect boxes in leaf using all threads
            findOveralppingsWId<NumThreadsPerDim>(scounts,
                                    selmIds,
                                    box,
                                    querySize,
                                    numBoxesInLeaf,
                                    sboxCache,
                                    sindCache,
                                    tid,
                                    treeIdx);
            __syncthreads();
            if(isValidBox)
                sumOverlappingPairs<NumThreadsPerDim>(outPairs,
                                    writeLoc,
                                    scounts,
                                    selmIds,
                                    numBoxesInLeaf,
                                    tid,
                                    queryIdx,
                                    boxInd);
            
            if(tid<1) {
// take out top of stack
                sstackSize--;
            }
            __syncthreads();
            
            if(sstackSize<1) break;		    
		}
		else {
            if(tid<1) {
                leftBox = internalNodeAabbs[getIndexWithInternalNodeMarkerRemoved(child.x)];
                b1 = isAabbOverlapping(groupBox, leftBox);
            }
            else if(tid==1) {
                rightBox = internalNodeAabbs[getIndexWithInternalNodeMarkerRemoved(child.y)];
                b2 = isAabbOverlapping(groupBox, rightBox);
            }
            __syncthreads();
            
            if(tid<1) {
                if(b1 == 0 && b2 == 0) {
// visit no child, take out top of stack
                    sstackSize--;
                }
                else if(b1 > b2) {
// visit right child
                    sstack[ sstackSize - 1 ] = child.x; 
                }
                else if(b2 > b1) {
// visit right child
                    sstack[ sstackSize - 1 ] = child.y; 
                }
                else {
// visit both children                    
                    sstack[ sstackSize - 1 ] = child.y;
                    if(sstackSize < BVH_TRAVERSE_MAX_STACK_SIZE) { 
                            sstack[ sstackSize ] = child.x;
                            sstackSize++;
                    }
                }
            }
            __syncthreads();
            
            if(sstackSize<1) break;
		}
	}
	if(isValidBox)
        cacheWriteLocation[boxInd] = writeLoc;
}
#endif        //  #ifndef OVERLAPPING2_CUH

