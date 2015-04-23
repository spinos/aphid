#ifndef SIMPLECONTACTSOLVER_IMPLEMENT_H
#define SIMPLECONTACTSOLVER_IMPLEMENT_H

#include "bvh_common.h"
#include "radixsort_implement.h"

struct ContactConstraint {
    BarycentricCoordinate coordA;
    BarycentricCoordinate coordB;
    float3 normal;
    float lambda;
    float Minv;
    float relVel;
};

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
                                        uint2 * splits,
                                        uint2 * pairs,
                                        float * mass,
	                                    uint4 * ind,
	                                    uint * perObjPointStart,
	                                    uint * perObjectIndexStart,
                                        uint * bodyCount, 
                                        uint bufLength);

void simpleContactSolverSetContactConstraint(ContactConstraint* constraints,
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

void simpleContactSolverSolveContact(ContactConstraint* constraints,
                        float3 * deltaLinearVelocity,
	                    float3 * deltaAngularVelocity,
	                    uint2 * pairs,
                        uint2 * splits,
	                    float * splitMass,
	                    ContactData * contacts,
	                    float3 * positions,
                        float3 * velocities,
                        uint4 * indices,
                        uint * perObjPointStart,
                        uint * perObjectIndexStart,
                        uint numContacts,
                        float * deltaJ,
                        int maxNIt,
                        int it);

void simpleContactSolverSolveContactWoJ(ContactConstraint* constraints,
                        float3 * deltaLinearVelocity,
	                    float3 * deltaAngularVelocity,
	                    uint2 * pairs,
                        uint2 * splits,
	                    float * splitMass,
	                    ContactData * contacts,
	                    float3 * positions,
                        float3 * velocities,
                        uint4 * indices,
                        uint * perObjPointStart,
                        uint * perObjectIndexStart,
                        uint numContacts);

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
                    ContactConstraint * constraints,
                    ContactData * contacts,
                    float3 * position,
                    uint4 * indices,
                    uint * objectPointStarts,
                    uint * objectIndexStarts,
                    uint numPoints);

}
#endif        //  #ifndef SIMPLECONTACTSOLVER_IMPLEMENT_H

