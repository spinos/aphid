#ifndef NARROWPHASE_IMPLEMENT_H
#define NARROWPHASE_IMPLEMENT_H

#include "bvh_common.h"

extern "C" {
    
void narrowphaseWriteObjectToCache(float3 * dstPos,
        float3 * dstVel,
        uint4 * dstInd,
        float3 * srcPos,
        float3 * srcVel,
        uint4 * srcInd,
        uint numPoints,
		uint numTetradedrons);

void narrowphaseComputeSeparateAxis(float4 * dstSA,
        float3 * dstPA, float3 * dstPB,
		uint2 * pairs,
		float3 * pos,
		float3 * vel,
		uint4 * ind,
		uint * pointStart, uint * indexStart, 
		uint numOverlappingPairs);
}
#endif        //  #ifndef NARROWPHASE_IMPLEMENT_H

