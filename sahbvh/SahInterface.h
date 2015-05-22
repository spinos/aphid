#ifndef SAHINTERFACE_H
#define SAHINTERFACE_H
#include "bvh_common.h"
#include "radixsort_implement.h"

#define SIZE_OF_SIMPLEQUEUE 32

namespace sahsplit {
    
void doSplitWorks(void * q, int * qelements,
                    int2 * nodes,
                    Aabb * nodeAabbs,
                    int * nodeParents,
                    int * nodeLevels,
                    KeyValuePair * primitiveIndirections,
                    Aabb * primitiveAabbs,
                    KeyValuePair * intermediateIndirections,
                    uint numPrimitives);

void initHash(KeyValuePair * primitiveIndirections,
                    uint numPrimitives);

}
#endif        //  #ifndef SAHINTERFACE_H

