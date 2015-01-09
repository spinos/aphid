#ifndef CREATEBVH_IMPLEMENT_H
#define CREATEBVH_IMPLEMENT_H

#include "bvh_common.h"

extern "C" void bvhCalculateAabbs(Aabb *dst, float4 * cvs, EdgeContact * edges, unsigned numEdges, unsigned numVertices);
extern "C" void bvhReduceAabb(Aabb *dst, Aabb *src, unsigned numAabbs, unsigned numBlocks, unsigned numThreads);
extern "C" void getReduceBlockThread(uint & blocks, uint & threads, uint n);

#endif        //  #ifndef CREATEBVH_IMPLEMENT_H

