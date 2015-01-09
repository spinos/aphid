#ifndef CREATEBVH_IMPLEMENT_H
#define CREATEBVH_IMPLEMENT_H

#include "bvh_common.h"

extern "C" void bvhCalculateAabbs(Aabb *dst, float4 * cvs, EdgeContact * edges, unsigned numEdges, unsigned numVertices);

#endif        //  #ifndef CREATEBVH_IMPLEMENT_H

