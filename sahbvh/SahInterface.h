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
                    uint numPrimitives);
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
}
#endif        //  #ifndef SAHINTERFACE_H

