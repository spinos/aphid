#ifndef NARROWPHASE_IMPLEMENT_H
#define NARROWPHASE_IMPLEMENT_H

#include "bvh_common.h"

extern "C" {
void narrowphaseComputeSeparateAxis(ContactData * dstContact,
		uint2 * pairs,
		float3 * pos,
		float3 * vel,
		uint4 * ind,
		uint * pointStart, uint * indexStart, 
		uint numOverlappingPairs);
}
#endif        //  #ifndef NARROWPHASE_IMPLEMENT_H

