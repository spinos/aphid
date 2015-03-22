#ifndef SIMPLECONTACTSOLVER_IMPLEMENT_H
#define SIMPLECONTACTSOLVER_IMPLEMENT_H

#include "bvh_common.h"
#include "radixsort_implement.h"

#define JACOBI_NUM_ITERATIONS 7

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

void simpleContactSolverSetContactConstraint(float3 * projLinVel,
                                        float3 * projAngVel,
                                        float * lambda,
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

void simpleContactSolverSolveContact(float * lambda,
                        float3 * linearVelocity,
	                    float3 * angularVelocity,
	                    uint2 * splits,
	                    float * splitMass,
                        ContactData * contacts,
                        float * deltaJ,
                        uint numContacts,
                        float3 * relV,
                        int it);
}
#endif        //  #ifndef SIMPLECONTACTSOLVER_IMPLEMENT_H

