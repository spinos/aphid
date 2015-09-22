#ifndef MASSSYSTEM_IMPL_H
#define MASSSYSTEM_IMPL_H

#include <bvh_common.h>

namespace masssystem {
void computeMass(float * dst,
                float * mass0,
                uint * anchored,
                float scale,
                uint maxInd);

void useAnchoredVelocity(float3 * vel, 
                float3 * anchoredVel,
                uint * anchor,
                uint maxInd);

void useAllAnchoredVelocity(float3 * vel, 
                float3 * anchoredVel,
                uint maxInd);

void integrate(float3 * pos,
                float3 * prePos,
                float3 * vel, 
                float3 * anchoredVel,
                uint * anchor,
                float dt, 
                uint maxInd);

void integrateAllAnchored(float3 * pos,
                    float3 * vel,
                    float3 * vela,
                    float dt,
                    uint maxInd);

void integrateSimple(float3 * pos, 
                float3 * vel, 
                float dt, 
                uint maxInd);

void addGravity(float3 * deltaVel,
                float * mass,
                float dt,
                uint maxInd);

void impulseForce(float3 * force,
                           float3 * deltaVel,
                           float * mass,
                           float dt,
                           uint maxInd);

void computeEnergy(float * dst,
                float * mass,
                float3 * vel,
                float defaultNodeMass,
                uint maxInd);

void computeLength(float * dst,
                float3 * vel,
                uint maxInd);

void zeroVelocity(float3 * vel,
                uint maxInd);

void setVelocity(float3 * deltaVel,
                float * mass,
                float x, float y, float z,
                uint maxInd);

}
#endif        //  #ifndef MASSSYSTEM_IMPL_H

