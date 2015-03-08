#ifndef SIMPLECONTACTSOLVER_IMPLEMENT_H
#define SIMPLECONTACTSOLVER_IMPLEMENT_H

#include "bvh_common.h"

extern "C" {
    void simpleContactSolverStopAtContact(float3 * dstVelocity,
                        uint2 * pairs,
                        uint4 * indices,
                        uint * objectPointStarts,
                        uint * objectIndexStarts,
                        uint numContacts);
}
#endif        //  #ifndef SIMPLECONTACTSOLVER_IMPLEMENT_H

