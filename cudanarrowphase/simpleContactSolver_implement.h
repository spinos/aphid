#ifndef SIMPLECONTACTSOLVER_IMPLEMENT_H
#define SIMPLECONTACTSOLVER_IMPLEMENT_H

#include "bvh_common.h"
#include "radixsort_implement.h"

#define JACOBI_NUM_ITERATIONS 8

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
                                        float * Minv,
                                        uint2 * splits,
                                        uint2 * pairs,
                                        float3 * pos,
                                        float3 * vel,
                                        uint4 * ind,
                                        uint * perObjPointStart,
                                        uint * perObjectIndexStart,
                                        float * splitMass,
                                        ContactData * contacts,
                                        uint numContacts);

void simpleContactSolverClearDeltaVelocity(float3 * deltaLinVel, 
                                        float3 * deltaAngVel, 
                                        uint bufLength);

void simpleContactSolverSolveContact(float * lambda,
                        float3 * deltaLinearVelocity,
	                    float3 * deltaAngularVelocity,
                        float3 * linearVelocity,
	                    float3 * angularVelocity,
	                    uint2 * splits,
	                    float * splitMass,
	                    float * Minv,
                        ContactData * contacts,
                        uint numContacts,
                        float * deltaJ,
                        int it);

void simpleContactSolverAverageVelocities(float3 * linearVelocity,
                        float3 * angularVelocity,
                        uint * bodyCount, 
                        KeyValuePair * srcInd,
                        uint numBodies);

void simpleContactSolverWritePointTetHash(KeyValuePair * pntTetHash,
	                uint2 * pairs,
	                uint2 * splits,
	                uint * bodyCount,
	                uint4 * ind,
	                uint * perObjPointStart,
                    uint * perObjectIndexStart,
	                uint numBodies,
	                uint bufLength);

void simpleContactSolverUpdateVelocity(float3 * dstVelocity,
                    float3 * deltaLinearVelocity,
	                float3 * deltaAngularVelocity,
	                KeyValuePair * pntTetHash,
                    uint2 * pairs,
                    uint2 * splits,
                    float3 * position,
                    uint4 * indices,
                    uint * objectPointStarts,
                    uint * objectIndexStarts,
                    uint numPoints);

}
#endif        //  #ifndef SIMPLECONTACTSOLVER_IMPLEMENT_H

