#ifndef SAHINTERFACE_H
#define SAHINTERFACE_H
#include "bvh_common.h"
#include "radixsort_implement.h"

#define SIZE_OF_SIMPLEQUEUE 32

namespace sahsplit {
    
int doSplitWorks(void * q, int * qelements,
                    int2 * nodes,
                    Aabb * nodeAabbs,
                    int * nodeParents,
                    int * nodeLevels,
                    KeyValuePair * primitiveIndirections,
                    Aabb * primitiveAabbs,
                    KeyValuePair * intermediateIndirections,
                    uint numPrimitives,
                    int initialNumNodes);
}

namespace sahcompress {
    
void computeRunHead(uint * runHeads,
                    KeyValuePair * morton,
                    uint d,
                    uint n,
                    uint scanLength);

void compressRunHead(uint * compressed, 
							uint * runHeads,
							uint * indices,
							uint n);

void computeRunHash(KeyValuePair * compressed, 
						KeyValuePair * morton,
						uint * indices,
                        uint m,
						uint d,
						uint n);

void computeRunLength(uint * runLength,
							uint * runHeads,
							uint nRuns,
							uint nPrimitives,
							uint bufLength);

void computeClusterAabbs(Aabb * clusterAabbs,
            Aabb * primitiveAabbs,
            uint * runHeads,
            uint * runLength,
            uint numRuns);

void computeSortedRunLength(uint * runLength,
							uint * runHeads,
							KeyValuePair * indirections,
							uint nRuns,
							uint nPrimitives,
							uint bufLength);

void computeSortedClusterAabbs(Aabb * clusterAabbs,
            KeyValuePair * primitiveIndirections,
            Aabb * primitiveAabbs,
            KeyValuePair * indirections,
            uint * runHeads,
            uint * runLength,
            uint numRuns);

void copyAabb(Aabb * dst, Aabb * src, uint n);
}

namespace sahdecompress {

/*
 *  i as value
 */
void initHash(KeyValuePair * primitiveIndirections,
                    uint numPrimitives);

/*
 *   set element lock to zero
 *   sum cluster length in each leaf node
 *   cluster length is runHead[i+1]-runHead[i]
 *   internal nodes has zero count
 */

void countLeaves(uint * leafLengths,
                    int * qelements,
                    int2 * nodes,
                    KeyValuePair * indirections,
                    uint * runHeads,
                    uint numHeads,
                    uint numPrimitives,
                    uint numNodes,
                    uint scanLength);

/*
 *  simple copy
 */
void copyHash(KeyValuePair * dst, KeyValuePair * src,
                uint n);
/*
 *   for each leaf node 
 *   replace cluster range with primitive range
 *   rearange primitive hash
 */
void decompressPrimitives(KeyValuePair * dst,
                            KeyValuePair * src,
                            int2 * nodes,
                            KeyValuePair * indirections,
                            uint* leafOffset,
                            uint * runHeads,
                            uint numHeads,
                            uint numPrimitives,
                            uint numNodes);

void decompressIndices(uint * decompressedIndices,
                    uint * compressedIndices,
					KeyValuePair * sorted,
					uint * offset,
					uint * runLength,
					uint n);

void rearrangeIndices(KeyValuePair * dst,
                        KeyValuePair * src,
                        uint * compressedIndices,
					KeyValuePair * sorted,
					uint * offset,
					uint * runLength,
					uint nunRuns);
}
#endif        //  #ifndef SAHINTERFACE_H

