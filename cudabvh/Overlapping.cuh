#ifndef OVERLAPPING_CUH
#define OVERLAPPING_CUH

#include "bvhUtil.h"

template<int NumExcls>
__device__ void writeElementExclusion(int * dst,
									uint a,
									int * exclusionInd)
{
    int i=0;
#if 0
    int4 * dstInd4 = (int4 *)dst;
	int4 * srcInd4 = (int4 *)&exclusionInd[NumExcls * a];
	for(;i<(NumExcls>>2); i++)
	    dstInd4[i] = srcInd4[i];
#else
    for(;i<NumExcls; i++)
	    dst[i] = exclusionInd[NumExcls * a + i];
#endif
}

template<int NumExcls>
__device__ int isElementExcludedS(int b, int * exclusionInd)
{
	uint i;
#if 0
    int4 * exclusionInd4 = (int4 *)exclusionInd;
	for(i=0; i<(NumExcls>>2); i++) {
		if(exclusionInd4[i].x < 0) break;
		if(b <= exclusionInd4[i].x) return 1;
		if(exclusionInd4[i].y < 0) break;
		if(b <= exclusionInd4[i].y) return 1;
		if(exclusionInd4[i].z < 0) break;
		if(b <= exclusionInd4[i].z) return 1;
		if(exclusionInd4[i].w < 0) break;
		if(b <= exclusionInd4[i].w) return 1;
	}
#else
    for(i=0; i<NumExcls; i++) {
		if(exclusionInd[i] < 0) break;
		if(b <= exclusionInd[i]) return 1;
	}
#endif
	return 0;
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

template<int NumExcls, int NumThreadsPerDim>
__global__ void countPairsSelfCollide_kernel(uint * overlappingCounts, 
								Aabb * boxes, 
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, 
								Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								int * exclusionIndices)
{
    int *sdata = SharedMemory<int>();
	
    uint queryInd = blockIdx.x;
    int2 queryRange = internalNodeChildIndices[queryInd];
    if(!isLeafNode(queryRange)) return;
    
	const uint tid = tId2();
	const int querySize = queryRange.y - queryRange.x + 1;
	const int isValidBox = (tid < querySize);
/* 
 *  smem layout in ints
 *  n as max num primitives in a leaf node 16
 *  m as max stack size 64
 *  e as exclusive size 32
 *
  *  0   -> 1      stackSize
 *  4   -> 4+m-1       stack
 *  4+m -> 4+m+n-1  visiting
 *  4+m+n -> 4+m+2n-1  visiting
 *  4+m+2n -> 4+m+2n+6*n-1  query boxes
 *  4+m+2n+6*n -> 4+m+2n+6*n+6*n-1  leaf boxes cache
 *  4+m+2n+12*n -> 4+m+2n+12*n+n-1  leaf ind cache
 *  4+m+2n+12*n+n -> 4+m+2n+12*n+n+e*n-1  box exclusive ind cache
 *  4+m+2n+12*n+n+e*n -> 4+m+2n+12*n+n+e*n+n*n-1 overlapping counts
 *  4+m+2n+12*n+n+e*n+n*n -> 4+m+2n+12*n+n+e*n+n*n+n*n-1 overlapping ids
 *
 *  visiting is n child to visit
 *  3 both 2 left 1 right 0 neither
 *  if max(visiting) == 0 pop stack
 *  if max(visiting) == 1 override top of stack by right
 *  if max(visiting) >= 2 override top of stack by right, then push left to stack
 */	
    int & sstackSize =          sdata[0];
    int * sstack =             &sdata[4];
    int * svisit =             &sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE];
    uint * squeryIdx = (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim];
    Aabb * squeryBox = (Aabb *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + 2 * NumThreadsPerDim];
    Aabb * sboxCache = (Aabb *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + 2 * NumThreadsPerDim + 6 * NumThreadsPerDim];
    uint * sindCache = (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + 2 * NumThreadsPerDim + 12 * NumThreadsPerDim];
    int * sexclusCache =       &sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + 2 * NumThreadsPerDim + 12 * NumThreadsPerDim + NumThreadsPerDim];
	int * scounts =            &sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + 2 * NumThreadsPerDim + 12 * NumThreadsPerDim + NumThreadsPerDim + NumExcls * NumThreadsPerDim];
	uint * selmIds =   (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + 2 * NumThreadsPerDim + 12 * NumThreadsPerDim + NumThreadsPerDim + NumExcls * NumThreadsPerDim + NumThreadsPerDim * NumThreadsPerDim]; 
	
    int * exclElm = &sexclusCache[NumExcls * threadIdx.x];
    if(isValidBox) {
       // writeElementExclusion<NumExcls>(exclElm, boxInd, exclusionIndices);
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
    int b1, b2;
	const uint boxInd = squeryIdx[threadIdx.x];
    const Aabb box = squeryBox[threadIdx.x];
    uint outCount = 0;
    
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
/*
            findOveralppings<NumExcls, NumThreadsPerDim>(scounts,
                                    box,
                                    querySize,
                                    numBoxesInLeaf,
                                    sboxCache,
                                    sindCache,
                                    &exclusionIndices[boxInd*NumExcls],
                                    tid);
*/                               
                        findOveralppingsWElmId<NumExcls, NumThreadsPerDim>(scounts,
                                    selmIds,
                                    box,
                                    querySize,
                                    numBoxesInLeaf,
                                    sboxCache,
                                    sindCache,
                                    //exclElm,
                                    &exclusionIndices[boxInd*NumExcls],
                                    tid,
                                    0);
            __syncthreads();
            if(isValidBox)
                sumOverlappingCounts<NumThreadsPerDim>(outCount, 
                                    scounts,
                                    numBoxesInLeaf,
                                    tid);
            __syncthreads();
            if(tid<1) {
// take out top of stack
                sstackSize--;
            }
            __syncthreads();
            
            if(sstackSize<1) break;
        }
        else {
            if(isValidBox) {
                leftBox = internalNodeAabbs[getIndexWithInternalNodeMarkerRemoved(child.x)];
                b1 = isAabbOverlapping(box, leftBox);
            
                rightBox = internalNodeAabbs[getIndexWithInternalNodeMarkerRemoved(child.y)];
                b2 = isAabbOverlapping(box, rightBox);

                svisit[tid] = 2 * b1 + b2;
            }
            else {
                svisit[tid] = 0;
            }
            __syncthreads();
            
            reduceMaxInBlock<NumThreadsPerDim, int>(tid, svisit);
		
            if(tid<1) {
                if(svisit[tid] == 0) {
// visit no child, take out top of stack
                    sstackSize--;
                }
                else if(svisit[tid] == 1) {
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

template<int NumExcls, int NumThreadsPerDim>
__global__ void writePairCacheSelfCollide_kernel(uint2 * dst, 
                                uint * cacheWriteLocation,
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
    const int querySize = queryRange.y - queryRange.x + 1;
	const int isValidBox = (tid < querySize);
/* 
 *  smem layout in ints
 *  n as max num primitives in a leaf node 16
 *  m as max stack size 64
 *  e as exclusive size 32
 *
 *  0   -> 1      stackSize
 *  4   -> 4+m-1       stack
 *  4+m -> 4+m+n-1  visiting
 *  4+m+n -> 4+m+2n-1  visiting
 *  4+m+2n -> 4+m+2n+6*n-1  query boxes
 *  4+m+2n+6*n -> 4+m+2n+6*n+6*n-1  leaf boxes cache
 *  4+m+2n+12*n -> 4+m+2n+12*n+n-1  leaf ind cache
 *  4+m+2n+12*n+n -> 4+m+2n+12*n+n+e*n-1  box exclusive ind cache
 *  4+m+2n+12*n+n+e*n -> 4+m+2n+12*n+n+e*n+n*n-1 overlapping counts
 *  4+m+2n+12*n+n+e*n+n*n -> 4+m+2n+12*n+n+e*n+n*n+n*n-1 overlapping ids
 */
    int & sstackSize =          sdata[0];
    int * sstack =             &sdata[4];
    int * svisit =             &sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE];
    uint * squeryIdx = (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreadsPerDim];
    Aabb * squeryBox = (Aabb *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + 2 * NumThreadsPerDim];
    Aabb * sboxCache = (Aabb *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + 2 * NumThreadsPerDim + 6 * NumThreadsPerDim];
    uint * sindCache = (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + 2 * NumThreadsPerDim + 12 * NumThreadsPerDim];
    int * sexclusCache =       &sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + 2 * NumThreadsPerDim + 12 * NumThreadsPerDim + NumThreadsPerDim];
	int * scounts =            &sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + 2 * NumThreadsPerDim + 12 * NumThreadsPerDim + NumThreadsPerDim + NumExcls * NumThreadsPerDim];
	uint * selmIds =   (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + 2 * NumThreadsPerDim + 12 * NumThreadsPerDim + NumThreadsPerDim + NumExcls * NumThreadsPerDim + NumThreadsPerDim * NumThreadsPerDim]; 
	
    int * exclElm = &sexclusCache[NumExcls * threadIdx.x];
    if(isValidBox) {
        // writeElementExclusion<NumExcls>(exclElm, boxInd, exclusionIndices);
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
    int b1, b2;
    
    const uint boxInd = squeryIdx[threadIdx.x];
    const Aabb box = squeryBox[threadIdx.x];
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
                                    //exclElm,
                                    &exclusionIndices[boxInd*NumExcls],
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
            __syncthreads();
            if(tid<1) {
// take out top of stack
                sstackSize--;
            }
            __syncthreads();
            
            if(sstackSize<1) break;
		}
		else {
		    if(isValidBox) {
                leftBox = internalNodeAabbs[getIndexWithInternalNodeMarkerRemoved(child.x)];
                b1 = isAabbOverlapping(box, leftBox);
            
                rightBox = internalNodeAabbs[getIndexWithInternalNodeMarkerRemoved(child.y)];
                b2 = isAabbOverlapping(box, rightBox);
            
                svisit[tid] = 2 * b1 + b2;
            }
            else {
                svisit[tid] = 0;
            }
            __syncthreads();
            
            reduceMaxInBlock<NumThreadsPerDim, int>(tid, svisit);
		
            if(tid<1) {
                if(svisit[tid] == 0) {
// visit no child, take out top of stack
                    sstackSize--;
                }
                else if(svisit[tid] == 1) {
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

