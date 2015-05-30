#ifndef BVHINTERFACE_H
#define BVHINTERFACE_H

#include "bvh_common.h"
namespace bvhcost {
void computeTraverseCost(float * costs,
        int2 * nodes,
	    Aabb * nodeAabbs,
        uint n);
}

namespace bvhlazy {
void updateNodeAabbAtLevel(Aabb * nodeAabbs,
        int * nodeLevels,
        int2 * nodes,
        KeyValuePair * primitiveIndirections,
        Aabb * primitiveAabbs,
        int level,
	    uint n);
}

#endif        //  #ifndef BVHINTERFACE_H

