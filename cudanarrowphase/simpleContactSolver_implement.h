#ifndef SIMPLECONTACTSOLVER_IMPLEMENT_H
#define SIMPLECONTACTSOLVER_IMPLEMENT_H

#include "bvh_common.h"
#include "radixsort_implement.h"

extern "C" {
void simpleContactSolverWriteContactIndex(KeyValuePair * dstInd, 
                                    uint * srcInd, 
                                    uint n, uint bufferLength);

void simpleContactSolverCountUniqueBody(uint * dstCount,
                                    KeyValuePair * srcInd, 
                                    uint num, uint bufLength);

void simpleContactSolverComputeSplitBufLoc(uint2 * splits, 
                                    uint2 * srcPairs, 
                                    KeyValuePair * bodyPairHash, 
                                    uint bufLength);

void simpleContactSolverStopAtContact(float3 * dstVelocity,
                    uint2 * pairs,
                    uint4 * indices,
                    uint * objectPointStarts,
                    uint * objectIndexStarts,
                    uint numContacts);
}
#endif        //  #ifndef SIMPLECONTACTSOLVER_IMPLEMENT_H

