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

void narrowphaseComputeTimeOfImpact(ContactData * dstContact,
		uint2 * pairs,
		float3 * pos,
		float3 * vel,
		uint4 * ind,
		uint * pointStart, uint * indexStart, 
		uint numOverlappingPairs);

void narrowphaseComputeValidPairs(uint * dstCounts, 
        ContactData * srcContact, 
        uint numContacts, 
        uint scanBufferLength);

void narrowphaseSqueezeContactPairs(uint2 * dstPairs, uint2 * srcPairs,
                                    ContactData * dstContact, ContactData *srcContact,
									uint * counts, uint * packLoc, 
									uint maxInd);
}
#endif        //  #ifndef NARROWPHASE_IMPLEMENT_H

