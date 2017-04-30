#ifndef REDUCEBOX_IMPLEMENT_H
#define REDUCEBOX_IMPLEMENT_H

/*
 *  reduceBox_implement.h
 *  cudabvh
 *
 *  Created by jian zhang on 1/10/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include "reduce_common.h"

extern "C" void bvhReduceAabbByPoints(Aabb *dst, float3 *src, unsigned numPoints, unsigned numBlocks, unsigned numThreads, unsigned maxPInd);
extern "C" void bvhReduceAabbByAabb(Aabb *dst, Aabb *src, unsigned numAabbs, unsigned numBlocks, unsigned numThreads);

#endif        //  #ifndef REDUCEBOX_IMPLEMENT_H

