#include <bvh_common.h>

namespace masssystem {
void computeMass(float * dst,
                float * mass0,
                uint * anchored,
                float scale,
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
}
