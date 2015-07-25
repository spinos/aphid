#ifndef TRIANGLESYSTEMINTERFACE_H
#define TRIANGLESYSTEMINTERFACE_H

#include "bvh_common.h"
namespace trianglesys {
void formTetrahedronAabbs(Aabb * leafAabbs,
                                float3 * pos,
                                float3 * vel,
                                float dt,
                                uint4 * tets,
                                uint n);

void integrate(float3 * pos,
                    float3 * vel,
                    float3 * vela,
                    float dt,
                    uint n);
}
#endif        //  #ifndef TRIANGLESYSTEMINTERFACE_H

