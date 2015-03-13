#ifndef SIMPLECONTACTSOLVER_IMPLEMENT_H
#define SIMPLECONTACTSOLVER_IMPLEMENT_H

#include "bvh_common.h"
#include "radixsort_implement.h"

extern "C" {
void simpleContactSolverWriteContactIndex(KeyValuePair * dstInd, 
                                    uint * srcInd, 
                                    uint n, uint bufferLength);

void simpleContactSolverCountBody(uint * dstCount,
                                    KeyValuePair * srcInd, 
                                    uint num);

void simpleContactSolverComputeSplitBufLoc(uint2 * splits, 
                                    uint2 * srcPairs, 
                                    KeyValuePair * bodyPairHash, 
                                    uint bufLength);

void simpleContactSolverComputeSplitInverseMass(float * invMass, 
                                        uint * bodyCount, 
                                        uint bufLength);

void simpleContactSolverSetContactConstraint(float3 * linVelA,
                                             float3 * linVelB,
                                        float3 * angVelA,
                                        float3 * angVelB,
                                        float * lambda,
                                        float * invMass, 
                                        float * splitInvMass, 
                                        uint2 * splits,
                                        uint2 * pairs,
                                        float3 * pos,
                                        float3 * vel,
                                        uint4 * ind,
                                        uint * perObjPointStart,
                                        uint * perObjectIndexStart,
                                        uint numContacts);
	

void simpleContactSolverClearDeltaVelocity(float3 * deltaLinVel, 
                                        float3 * deltaAngVel, 
                                        uint bufLength);

void simpleContactSolverStopAtContact(float3 * dstVelocity,
                    uint2 * pairs,
                    uint4 * indices,
                    uint * objectPointStarts,
                    uint * objectIndexStarts,
                    uint numContacts);

void simpleContactSolverSolveContact(float3 * deltaLinVel,
	                    float3 * deltaAngVel,
	                    uint2 * splits,
	                    float * splitMass,
	                    float3 * srcVelocity,
                    uint2 * pairs,
                    uint4 * indices,
                    uint * objectPointStarts,
                    uint * objectIndexStarts,
                    uint numContacts);
}
#endif        //  #ifndef SIMPLECONTACTSOLVER_IMPLEMENT_H

