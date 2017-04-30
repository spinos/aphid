#include <bvh_common.h>

namespace windforce {
    
void setWindVecs(float * u,
                float * v,
                float * w);

void setWind(float3 * deltaVel,
                float * mass,
                float turbulence,
                uint windSeed,
                uint maxInd);

}

namespace gravityforce {
    
void setGravity(float * g);

void addGravity(float3 * deltaVel,
                float * mass,
                float dt,
                uint maxInd);

}
