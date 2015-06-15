#ifndef OVERLAPPING_CUH
#define OVERLAPPING_CUH

#include "bvhUtil.h"

template<int NumExcls>
__device__ void writeElementExclusion(int * dst,
									uint a,
									int * exclusionInd)
{
    int4 * dstInd4 = (int4 *)dst;
	int4 * srcInd4 = (int4 *)&exclusionInd[NumExcls * a];
	int i=0;
	for(;i<(NumExcls>>2); i++)
	    dstInd4[i] = srcInd4[i];
}

template<int NumExcls>
__device__ int isElementExcludedS(uint b, int * exclusionInd)
{
	uint i;
#if 1
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

template<int NumExcls>
inline __device__ void countOveralppingsExclS(uint & count,
                                         Aabb box,
                                         int n,
                                         Aabb * elementBoxes,
                                         uint * elementInd,
                                         int * exclElm)
{
    uint iElement;
    int i=0;
    for(;i<n;i++) {
        iElement = elementInd[i];
        if(isElementExcludedS<NumExcls>(iElement, exclElm)) continue;
        if(isAabbOverlapping(box, elementBoxes[i])) 
            count++;
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
    uint iElement;
    int i=0;
    for(;i<n;i++) {
        iElement = elementInd[i];
        if(isElementExcludedS<NumExcls>(iElement, exclElm)) continue;
        if(isAabbOverlapping(box, elementBoxes[i])) {
            pair.x = combineObjectElementInd(iQuery, iBox);
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
    uint iElement;
    int i=range.x;
    for(;i<=range.y;i++) {
        iElement = elementHash[i].value;
        if(isElementExcludedS<NumExcls>(iElement, exclElm)) continue;
        if(isAabbOverlapping(box, elementBoxes[iElement])) {
            pair.x = combineObjectElementInd(iQuery, iBox);
			pair.y = combineObjectElementInd(iQuery, iElement);
			overlappings[writeLoc] = pair;
            writeLoc++;
        }
    }
}

template<int NumExcls, int NumThreads>
__global__ void countPairsSelfCollide_kernel(uint * overlappingCounts, 
								Aabb * boxes, 
								KeyValuePair * queryIndirection,
                                uint maxBoxInd,
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, 
								Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								int * exclusionIndices)
{
    int *sdata = SharedMemory<int>();
	
	uint ind = blockIdx.x*blockDim.x + threadIdx.x;
	const int isValidBox = (ind < maxBoxInd);
	
	uint boxInd;
	Aabb box;
    if(isValidBox) {
        boxInd = queryIndirection[ind].value;
        box = boxes[boxInd];
    }
/* 
 *  smem layout in ints
 *  n as num threads    64
 *  m as max stack size 64
 *  c as box cache size 64
 *  e as exclusive size 32
 *
 *  0   -> 1      stackSize
 *  4   -> 4+m-1       stack
 *  4+m -> 4+m+n-1  visiting
 *  4+m+n -> 4+m+n+6*c-1  leaf boxes cache
 *  4+m+n+6*c -> 4+m+n+6*c+c-1  leaf ind cache
 *  4+m+n+6*c+c -> 4+m+n+6*c+c+e*n-1  box exclusive ind cache
 *
 *  visiting is n child to visit
 *  3 both 2 left 1 right 0 neither
 *  if max(visiting) == 0 pop stack
 *  if max(visiting) == 1 override top of stack by right
 *  if max(visiting) >= 2 override top of stack by right, then push left to stack
 */	
    int & sstackSize = sdata[0];
    int * sstack =  &sdata[4];
    int * svisit = &sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE];
    Aabb * sboxCache = (Aabb *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreads];
    uint * sindCache = (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreads + BVH_PACKET_TRAVERSE_CACHE_SIZE * 6];
    int * sexclusCache = &sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreads + BVH_PACKET_TRAVERSE_CACHE_SIZE * 6 + BVH_PACKET_TRAVERSE_CACHE_SIZE];
	
    const uint tid = threadIdx.x;
    int * exclElm = &sexclusCache[tid*NumExcls];
    if(isValidBox) 
        writeElementExclusion<NumExcls>(exclElm, boxInd, exclusionIndices);
    
    if(tid<1) {
        sstack[0] = 0x80000000;
        sstackSize = 1;
    }
    __syncthreads();
		
	int canLeafFitInSmem;
    int numBoxesInLeaf;
	int isLeaf;
    int iNode;
    int2 child;
    Aabb leftBox, rightBox;
    int b1, b2;
	
    uint outCount = 0;
	for(;;) {
		iNode = sstack[ sstackSize - 1 ];
		iNode = getIndexWithInternalNodeMarkerRemoved(iNode);
        child = internalNodeChildIndices[iNode];
        isLeaf = iLeafNode(child);

        if(isLeaf) {
// load leaf boxes into smem
            numBoxesInLeaf = child.y - child.x + 1;
            canLeafFitInSmem = (numBoxesInLeaf <= BVH_PACKET_TRAVERSE_CACHE_SIZE);
            if(canLeafFitInSmem) 
                putLeafBoxAndIndInSmem<NumThreads>(sboxCache,
                                            sindCache,
                                            tid,
                                            numBoxesInLeaf,
                                            child,
                                            mortonCodesAndAabbIndices,
                                            leafAabbs);
            __syncthreads(); 
            if(isValidBox) {
// intersect boxes in leaf
            if(canLeafFitInSmem)
                countOveralppingsExclS<NumExcls>(outCount, 
                                    box,
                                    numBoxesInLeaf,
                                    sboxCache,
                                    sindCache,
                                    exclElm);
            else
                countOveralppingsExclG<NumExcls>(outCount,
                                    box,
                                    child,
                                    mortonCodesAndAabbIndices,
                                    leafAabbs,
                                    exclElm);
            }
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
            
            reduceMaxInBlock<NumThreads, int>(tid, svisit);
		
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

template<int NumExcls, int NumThreads>
__global__ void writePairCacheSelfCollide_kernel(uint2 * dst, 
                                uint * cacheWriteLocation,
								Aabb * boxes, 
								KeyValuePair * queryIndirection,
                                uint maxBoxInd,
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, 
								Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								unsigned queryIdx,
								int * exclusionIndices)
{
    int *sdata = SharedMemory<int>();
	
	uint ind = blockIdx.x*blockDim.x + threadIdx.x;
	const int isValidBox = (ind < maxBoxInd);
	
	uint boxInd;
	uint writeLoc;
    Aabb box;
    if(isValidBox) {
         boxInd = queryIndirection[ind].value;
	     writeLoc = cacheWriteLocation[boxInd];
         box = boxes[boxInd];	
	}
/* 
 *  smem layout in ints
 *  n as num threads    64
 *  m as max stack size 64
 *  c as box cache size 64
 *  e as exclusive size 32
 *
 *  0   -> 1      stackSize
 *  4   -> 4+m-1       stack
 *  4+m -> 4+m+n-1  visiting
 *  4+m+n -> 4+m+n+6*c-1  leaf boxes cache
 *  4+m+n+6*c -> 4+m+n+6*c+c-1  leaf ind cache
 *  4+m+n+6*c+c -> 4+m+n+6*c+c+e*n-1  box exclusive ind cache
 *
 *  visiting is n child to visit
 *  3 both 2 left 1 right 0 neither
 *  if max(visiting) == 0 pop stack
 *  if max(visiting) == 1 override top of stack by right
 *  if max(visiting) >= 2 override top of stack by right, then push left to stack
 */
    int & sstackSize = sdata[0];
    int * sstack =  &sdata[4];
    int * svisit = &sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE];
    Aabb * sboxCache = (Aabb *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreads];
    uint * sindCache = (uint *)&sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreads + BVH_PACKET_TRAVERSE_CACHE_SIZE * 6];
    int * sexclusCache = &sdata[4 + BVH_TRAVERSE_MAX_STACK_SIZE + NumThreads + BVH_PACKET_TRAVERSE_CACHE_SIZE * 6 + BVH_PACKET_TRAVERSE_CACHE_SIZE];
	const uint tid = threadIdx.x;
    int * exclElm = &sexclusCache[tid*NumExcls];
    if(isValidBox) 
        writeElementExclusion<NumExcls>(exclElm, boxInd, exclusionIndices);
    
	if(tid<1) {
        sstack[0] = 0x80000000;
        sstackSize = 1;
    }
    __syncthreads();
		
	int canLeafFitInSmem;
    int numBoxesInLeaf;
	int isLeaf;
    int iNode;
    int2 child;
    Aabb leftBox, rightBox;
    int b1, b2;
	
	for(;;) {
		iNode = sstack[ sstackSize - 1 ];
		iNode = getIndexWithInternalNodeMarkerRemoved(iNode);
        child = internalNodeChildIndices[iNode];
        isLeaf = iLeafNode(child);
		
		if(isLeaf) {
// load leaf boxes into smem
            numBoxesInLeaf = child.y - child.x + 1;
            canLeafFitInSmem = (numBoxesInLeaf <= BVH_PACKET_TRAVERSE_CACHE_SIZE);
            if(canLeafFitInSmem) 
                putLeafBoxAndIndInSmem<NumThreads>(sboxCache,
                                            sindCache,
                                            tid,
                                            numBoxesInLeaf,
                                            child,
                                            mortonCodesAndAabbIndices,
                                            leafAabbs);
            __syncthreads();
            if(isValidBox) {
// intersect boxes in leaf
            if(canLeafFitInSmem)
                writeOverlappingsExclS<NumExcls>(dst,
                                    writeLoc,
                                    queryIdx,
                                    boxInd,
                                    box,
                                    numBoxesInLeaf,
                                    sboxCache,
                                    sindCache,
                                    exclElm);
            else
                writeOverlappingsExclG<NumExcls>(dst,
                                    writeLoc,
                                    queryIdx,
                                    boxInd,
                                    box,
                                    child,
                                    mortonCodesAndAabbIndices,
                                    leafAabbs,
                                    exclElm);
            }
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
            
            reduceMaxInBlock<NumThreads, int>(tid, svisit);
		
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

