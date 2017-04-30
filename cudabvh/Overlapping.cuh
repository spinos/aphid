#ifndef OVERLAPPING_CUH
#define OVERLAPPING_CUH

/*
 *  self-collision packet traverse
 */
 
#include "bvhUtil.cuh"

template<int NumExcls>
inline __device__ void countOverlappings1(uint & count,
                                         uint iBox,
                                         Aabb box,
                                         Aabb * elementBoxes,
                                         uint * elementInds,
                                         int * exclElm,
                                         int n)
{
    uint iElement;
    int i= 0;
    for(;i<n;i++) {
        iElement = elementInds[i];
        if(iElement > iBox) {
            if(!isElementExcludedS<NumExcls>(iElement, exclElm)) {
                if(isAabbOverlapping(box, elementBoxes[i])) 
                    count++;
            }
        }
    }
}

template<int NumExcls, int NumThreadsPerDim>
__device__ void findOveralppings(int * scounts,
                                         Aabb box,
                                         int m,
                                         int n,
                                         Aabb * elementBoxes,
                                         uint * elementInd,
                                         int * exclElm,
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
    uint iElement;
    if(threadIdx.x < m && threadIdx.y < n) {
        iElement = elementInd[threadIdx.y];
        if(!isElementExcludedS<NumExcls>(iElement, exclElm)) {
            if(isAabbOverlapping(box, elementBoxes[threadIdx.y])) 
                scounts[tid] = 1;
        }
    }
}

template<int NumExcls, int NumThreadsPerDim>
__device__ void findOveralppingsWElmId(int * scounts,
                                        uint * selmIds,
                                         Aabb box,
                                         int m,
                                         int n,
                                         Aabb * elementBoxes,
                                         uint * elementInd,
                                         int * exclElm,
                                         uint tid,
                                         uint iQuery)
{
    scounts[tid] = 0;
    uint iElement;
    if(threadIdx.x < m && threadIdx.y < n) {
        iElement = elementInd[threadIdx.y];
        if(!isElementExcludedS<NumExcls>(iElement, exclElm)) {
            if(isAabbOverlapping(box, elementBoxes[threadIdx.y])) {
                scounts[tid] = 1;
                selmIds[tid] = combineObjectElementInd(iQuery, iElement);
            }
        }
    }
}

template<int NumThreadsPerDim>
__device__ void sumOverlappingCounts(uint & outCount,
                                int * scounts,
                                int n,
                                uint tid)
{
    int i = 0;
    for(;i<n;i++) {
        if(scounts[tid + NumThreadsPerDim * i])
            outCount++;
    }
}

template<int NumThreadsPerDim>
__device__ void sumOverlappingPairs(uint2 * overlappings,
                                uint & writeLoc,
                                int * scounts,
                                uint * selmIds,
                                int n,
                                uint tid,
                                uint iQuery,
                                uint iBox)
{
    uint2 pair;
    pair.x = combineObjectElementInd(iQuery, iBox);
    uint loc;
    int i = 0;
    for(;i<n;i++) {
        loc = tid + NumThreadsPerDim * i;
        if(scounts[loc]) {
            pair.y = selmIds[loc];
			overlappings[writeLoc] = pair;
            writeLoc++;
        }
    }
}

template<int NumExcls>
inline __device__ void countOveralppingsExclG(uint & count,
                                         Aabb box,
                                         int2 range,
                                         KeyValuePair * elementHash,
                                         Aabb * elementBoxes,
                                         int * exclElm)
{
    uint iElement;
    int i=range.x;
    for(;i<=range.y;i++) {
        iElement = elementHash[i].value;
        if(isElementExcludedS<NumExcls>(iElement, exclElm)) continue;
        if(isAabbOverlapping(box, elementBoxes[iElement])) 
            count++;
    }
}

template<int NumExcls>
inline __device__ void writeOverlappingsExclS(uint2 * overlappings,
                                uint & writeLoc,
                                uint iQuery,
                                uint iBox,
                                Aabb box,
                                         int n,
                                         Aabb * elementBoxes,
                                         uint * elementInd,
                                         int * exclElm)
{
    uint2 pair;
    pair.x = combineObjectElementInd(iQuery, iBox);
	uint iElement;
    int i=0;
    for(;i<n;i++) {
        iElement = elementInd[i];
        if(isElementExcludedS<NumExcls>(iElement, exclElm)) continue;
        if(isAabbOverlapping(box, elementBoxes[i])) {
            pair.y = combineObjectElementInd(iQuery, iElement);
			overlappings[writeLoc] = pair;
            writeLoc++;
        }
    }
}

template<int NumExcls>
inline __device__ void writeOverlappingsExclG(uint2 * overlappings,
                                uint & writeLoc,
                                uint iQuery,
                                uint iBox,
                                Aabb box,
                                int2 range,
                                KeyValuePair * elementHash,
                                Aabb * elementBoxes,
                                int * exclElm)
{
    uint2 pair;
    pair.x = combineObjectElementInd(iQuery, iBox);
    uint iElement;
    int i=range.x;
    for(;i<=range.y;i++) {
        iElement = elementHash[i].value;
        if(isElementExcludedS<NumExcls>(iElement, exclElm)) continue;
        if(isAabbOverlapping(box, elementBoxes[iElement])) {
            pair.y = combineObjectElementInd(iQuery, iElement);
			overlappings[writeLoc] = pair;
            writeLoc++;
        }
    }
}

template<int NumExcls>
__global__ void countPairsSelfCollidePacket_kernel(uint * overlappingCounts, 
								Aabb * boxes,
								uint numInternalNodes,
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, 
								Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								int * exclusionIndices)
{
    int *sdata = SharedMemory<int>();
// one node for each warp, 8 warps per block
    uint queryInd = blockIdx.x * 8 + wId();
    if(queryInd >= numInternalNodes) return;
// for leaf node only
    int2 queryRange = internalNodeChildIndices[queryInd];
    if(!isLeafNode(queryRange)) return;
    
    const int querySize = queryRange.y - queryRange.x + 1;
	
    const uint iWarp = wId();
    const uint tWarp = tIdW();
	const int isValidBox = (tWarp < querySize);
/* 
 *  smem layout in ints
 *  m as max stack size 64
 *  e as exclusive size 32
 *
 *  0   -> 7     warp stackSize
 *  8   -> 8+8m-1   warp stack
 *  8+8m -> 8+8m+48-1 wart group box
 *  8+8m+48 -> 8+8m+48+48-1 warp internal box
 *  8+8m+96 -> 8+8m+96+8-1 warp iNode
 *  8+8m+104 ->8+8m+104+8-1 warp isInternal
 *  8+8m+112 ->8+8m+112+8-1 warp isOverlap
 *  8+8m+120 ->8+8m+120+16-1 warp child
 *  8+8m+136 -> 8+8m+136+8*6*32-1 warp element box
 *  8+8m+1672 -> 8+8m+1672+8*32-1 warp element ind
 */	
    int & sstackSize =              sdata[iWarp];
    int * sstack =                  &sdata[8 + iWarp * BVH_TRAVERSE_MAX_STACK_SIZE];
    Aabb * sgroupBox = (Aabb *)      &sdata[8 + 8 * BVH_TRAVERSE_MAX_STACK_SIZE + iWarp * 6];
    Aabb * sinternalBox = (Aabb *)   &sdata[8 + 8 * BVH_TRAVERSE_MAX_STACK_SIZE + 8*6 + iWarp * 6];
    int & iNode = sdata[8 + 8 * BVH_TRAVERSE_MAX_STACK_SIZE + 96 + iWarp];
    int & isInternal = sdata[8 + 8 * BVH_TRAVERSE_MAX_STACK_SIZE + 104 + iWarp];
    int & isOverlap = sdata[8 + 8 * BVH_TRAVERSE_MAX_STACK_SIZE + 112 + iWarp];
    int2 * child = (int2 *)&sdata[8 + 8 * BVH_TRAVERSE_MAX_STACK_SIZE + 120 + iWarp * 2];
    Aabb * selmBox = (Aabb *)&sdata[8 + 8 * BVH_TRAVERSE_MAX_STACK_SIZE + 136 + iWarp * 32*6];
    uint * selmInd = (uint *)&sdata[8 + 8 * BVH_TRAVERSE_MAX_STACK_SIZE + 1672 + iWarp * 32];

    if(tWarp < 1) {
        *sgroupBox = internalNodeAabbs[queryInd];
        sstack[0] = 0x80000000;
        sstackSize = 1;
    }
    __syncthreads();
    
    int *exclElm; 
    uint boxInd;
    Aabb box;
    if(isValidBox) {
        boxInd = mortonCodesAndAabbIndices[queryRange.x + tWarp].value;
        box = boxes[boxInd];
        exclElm = &exclusionIndices[NumExcls * boxInd];
    }
    
	uint outCount = 0;
    
	for(;;) {
	    if(sstackSize < 1) break;
		
	    if(tWarp < 1) {
            iNode = sstack[ sstackSize - 1 ];
            sstackSize--;
		
            iNode = getIndexWithInternalNodeMarkerRemoved(iNode);
            *child = internalNodeChildIndices[iNode];
            isInternal = isInternalNode(*child);
            
            *sinternalBox = internalNodeAabbs[iNode];
            isOverlap = isAabbOverlapping(*sgroupBox, *sinternalBox);
        }
        __syncthreads();
        
        if(isOverlap) {    
		    if(!isInternal) {
		        putLeafBoxAndIndInSmem(selmBox,
		            selmInd,
		            tWarp,
		            child->y - child->x +1,
		            *child,
		            mortonCodesAndAabbIndices,
		            leafAabbs);
		        
		        if(isValidBox)
		            countOverlappings1<NumExcls>(outCount,
                                boxInd,
                                box,
                                selmBox,
                                selmInd,
                                exclElm,
                                child->y - child->x +1);
            }
            else {
                if(tWarp < 1) {
                    if(sstackSize >= BVH_TRAVERSE_MAX_STACK_SIZE-2) continue;
                    
                    sstack[ sstackSize ] = child->x;
                    sstackSize++;
                    sstack[ sstackSize ] = child->y;
                    sstackSize++;
                }
                __syncthreads();
            }
		}
	}
	
	if(isValidBox)
	    overlappingCounts[boxInd] = outCount;
}

template<int NumExcls, int NumThreadsPerDim>
__global__ void writePairCacheSelfCollidePacket_kernel(uint2 * dst, 
                                uint * cacheWriteLocation,
                                uint * cacheSize,
								Aabb * boxes,
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, 
								Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								unsigned queryIdx,
								int * exclusionIndices)
{
    int *sdata = SharedMemory<int>();
	
    uint queryInd = blockIdx.x;
    int2 queryRange = internalNodeChildIndices[queryInd];
    if(!isLeafNode(queryRange)) return;
    
	const uint tid = tId2();
	int querySize = queryRange.y - queryRange.x + 1;
	//if(querySize > NumThreadsPerDim) querySize = NumThreadsPerDim;
	const int isValidBox = (tid < querySize);
/* 
 *  smem layout in ints
 *  n as max num primitives in a leaf node 16
 *  m as max stack size 64
 *  e as exclusive size 32
 *
 *  0   ->        stackSize
 *  1   ->       visit left child
 *  2   ->       visit right child
 *  4   -> 4+m-1       stack
 *  4+m -> 4+m+n-1  query index
 *  4+m+n+6n -> 4+m+n+6n+6n-1  leaf boxes cache
 *  4+m+n+12n -> 4+m+n+12n+n-1  leaf ind cache
 *  4+m+n+12n+n -> 4+m+n+12n+n+en-1  box exclusive ind cache
 *  4+m+n+12n+n+en -> 4+m+n+12n+n+en+nn-1 overlapping counts
 *  4+m+n+12n+n+en+nn -> 4+m+n+12n+n+en+nn+nn-1 overlapping ids
 *  4+m+n+12n+n+en+nn+nn -> 4+m+n+12n+n+en+nn+nn+nn-1  cache size
 */
    int & sstackSize =          sdata[0];
    int & b1 =                  sdata[1];
    int & b2 =                  sdata[2];
    int * sstack =             &sdata[4];
    uint * squeryIdx = (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE];
    Aabb * squeryBox = (Aabb *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim];
    Aabb * sboxCache = (Aabb *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim + 6 * NumThreadsPerDim];
    uint * sindCache = (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim + 12 * NumThreadsPerDim];
    int * sexclusCache =       &sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim + 12 * NumThreadsPerDim + NumThreadsPerDim];
	int * scounts =            &sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim + 12 * NumThreadsPerDim + NumThreadsPerDim + NumExcls * NumThreadsPerDim];
	uint * selmIds =   (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim + 12 * NumThreadsPerDim + NumThreadsPerDim + NumExcls * NumThreadsPerDim + NumThreadsPerDim * NumThreadsPerDim]; 
	//int * svisit =             &sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim + 12 * NumThreadsPerDim + NumThreadsPerDim + NumExcls * NumThreadsPerDim + 2 * NumThreadsPerDim * NumThreadsPerDim];
    
    if(isValidBox) {
        squeryIdx[tid] = mortonCodesAndAabbIndices[queryRange.x + tid].value;
        squeryBox[tid] = boxes[squeryIdx[tid]];
        //svisit[tid] = cacheSize[squeryIdx[tid]];
    //}
    //else 
      //  svisit[tid] = 0;
    
    //__syncthreads();
    //reduceMaxInBlock<NumThreadsPerDim, int>(tid, svisit);
	//__syncthreads();
	
	//if(svisit[0]<1) return;
	
	//if(isValidBox) {
        writeElementExclusion<NumExcls>(&sexclusCache[NumExcls*threadIdx.x], &exclusionIndices[NumExcls*squeryIdx[tid]]);
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
            findOveralppingsWElmId<NumExcls, NumThreadsPerDim>(scounts,
                                    selmIds,
                                    box,
                                    querySize,
                                    numBoxesInLeaf,
                                    sboxCache,
                                    sindCache,
                                    &sexclusCache[NumExcls*threadIdx.x],
                                    tid,
                                    queryIdx);
            __syncthreads();
            if(isValidBox)
                sumOverlappingPairs<NumThreadsPerDim>(dst,
                                    writeLoc,
                                    scounts,
                                    selmIds,
                                    numBoxesInLeaf,
                                    tid,
                                    queryIdx,
                                    boxInd);
            //__syncthreads();
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

__global__ void startAsWriteLocation_kernel(uint * dst, uint * src, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind < maxInd) dst[ind] = src[ind];
}

#endif        //  #ifndef OVERLAPPING_CUH

