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
}
