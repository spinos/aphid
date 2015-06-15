#ifndef OVERLAPPING2_CUH
#define OVERLAPPING2_CUH

#include "bvhUtil.h"

template<int NumThreads>
__device__ void putLeafBoxesInSmem(Aabb * dst,
                                   uint tid,
                                   uint n,
                                   int2 range,
                                   KeyValuePair * elementHash,
                                   Aabb * elementAabbs)
{
    uint iElement; 
    uint loc = tid;
    if(loc < n) {
        iElement = elementHash[range.x + loc].value;
        dst[loc] = elementAabbs[iElement];
    }
    
    if(n>NumThreads) {
        loc += NumThreads;
        if(loc < n) {
            iElement = elementHash[range.x + loc].value;
            dst[loc] = elementAabbs[iElement];
        }
    }
}

__device__ void countOveralppingsS(uint & count, 
                                    Aabb box,
                                    int n,
                                    Aabb * elementBoxes)
{
    int i=0;
    for(;i<n;i++) {
        if(isAabbOverlapping(box, elementBoxes[i])) 
            count++;
    }
}

__device__ void countOveralppingsG(uint & count,
                                    Aabb box,
                                    int2 range,
                                    KeyValuePair * elementHash,
                                    Aabb * elementBoxes)
{
    uint iElement;
    int i=range.x;
    for(;i<=range.y;i++) {
        iElement = elementHash[i].value;
        if(isAabbOverlapping(box, elementBoxes[iElement])) 
            count++;
    }
}

__device__ void writeOveralppingsS(uint2 * outPairs,
                                uint & writeLoc,
                                uint iQuery,
                                uint iBox,
                                Aabb box,
                                uint iTree,
                                int n,
                                Aabb * elementBoxes,
                                uint * elementInd,
                                const uint & startLoc,
                                const uint & cacheSize)
{
    uint2 pair;
    int i=0;
    for(;i<n;i++) {
        if(isAabbOverlapping(box, elementBoxes[i])) 
        {
            pair.x = combineObjectElementInd(iQuery, iBox);
			pair.y = combineObjectElementInd(iTree, elementInd[i]);
			outPairs[writeLoc] = pair;
            writeLoc++;
        }
        if((writeLoc - startLoc)==cacheSize) return;
    }
}


__device__ void writeOveralppingsG(uint2 * outPairs,
                                uint & writeLoc,
                                uint iQuery,
                                uint iBox,
                                Aabb box,
                                uint iTree,
                                int2 range,
                                KeyValuePair * elementHash,
                                Aabb * elementBoxes,
                                const uint & startLoc,
                                const uint & cacheSize)
{
    uint2 pair;
    uint iElement;
    int i=range.x;
    for(;i<=range.y;i++) {
        iElement = elementHash[i].value;
        if(isAabbOverlapping(box, elementBoxes[iElement])) 
        {
            pair.x = combineObjectElementInd(iQuery, iBox);
			pair.y = combineObjectElementInd(iTree, iElement);
			outPairs[writeLoc] = pair;
            writeLoc++;
        }
        if((writeLoc - startLoc)==cacheSize) return;
    }
}

template<int NumThreads>
__global__ void countPairs_kernel(uint * overlappingCounts, Aabb * boxes,
                                uint maxBoxInd,
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, 
								Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices)
{
    int *sdata = SharedMemory<int>();
    
	uint boxInd = blockIdx.x*blockDim.x + threadIdx.x;
    const int isValidBox = (boxInd < maxBoxInd);
	
	Aabb box;
    uint outCount;
    if(isValidBox) {
        box = boxes[boxInd];
        outCount = overlappingCounts[boxInd];
    }
/* 
 *  smem layout in ints
 *  n as num threads    64
 *  m as max stack size 64
 *  c as box cache size 64
 *
 *  0   -> 1      stackSize
 *  4   -> 4+m-1       stack
 *  4+m -> 4+m+n-1  visiting
 *  4+m+n -> 4+m+n+6*c-1  leaf boxes cache
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
    const uint tid = threadIdx.x;
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
                putLeafBoxesInSmem<NumThreads>(sboxCache,
                                            tid,
                                            numBoxesInLeaf,
                                            child,
                                            mortonCodesAndAabbIndices,
                                            leafAabbs);
            __syncthreads();
            
            if(isValidBox) {
// intersect boxes in leaf
            if(canLeafFitInSmem)
                countOveralppingsS(outCount, 
                                    box,
                                    numBoxesInLeaf,
                                    sboxCache);
            else
                countOveralppingsG(outCount,
                                    box,
                                    child,
                                    mortonCodesAndAabbIndices,
                                    leafAabbs);
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

template<int NumThreads>
__global__ void writePairCache_kernel(uint2 * outPairs, 
                                uint * cacheWriteLocation,
                                uint * cacheStarts, 
                                uint * overlappingCounts, 
                                Aabb * boxes,
                                uint maxBoxInd,
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								uint queryIdx, 
								uint treeIdx)
{
    int *sdata = SharedMemory<int>();
    
	uint boxInd = blockIdx.x*blockDim.x + threadIdx.x;
	const int isValidBox = (boxInd < maxBoxInd);
	
    uint cacheSize, startLoc, writeLoc;
    Aabb box;
    if(isValidBox) {
	     cacheSize = overlappingCounts[boxInd];
	     startLoc = cacheStarts[boxInd];
	     writeLoc = cacheWriteLocation[boxInd];
         box = boxes[boxInd];	
	}
/* 
 *  smem layout in ints
 *  n as num threads    64
 *  m as max stack size 64
 *  c as box cache size 64
 *
 *  0   -> 1      stackSize
 *  4   -> 4+m-1       stack
 *  4+m -> 4+m+n-1  visiting
 *  4+m+n -> 4+m+n+6*c-1  leaf boxes cache
 *  4+m+n+6*c -> 4+m+n+6*c+c-1 leaf boxes ind
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
    const uint tid = threadIdx.x;
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
                writeOveralppingsS(outPairs, 
                                    writeLoc,
                                    queryIdx,
                                    boxInd,
                                    box,
                                    treeIdx,
                                    numBoxesInLeaf,
                                    sboxCache,
                                    sindCache,
                                    startLoc,
                                    cacheSize);
            else
                writeOveralppingsG(outPairs,
                                    writeLoc,
                                    queryIdx,
                                    boxInd,
                                    box,
                                    treeIdx,
                                    child,
                                    mortonCodesAndAabbIndices,
                                    leafAabbs,
                                    startLoc,
                                    cacheSize);
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
#endif        //  #ifndef OVERLAPPING2_CUH

