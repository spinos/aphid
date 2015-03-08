#ifndef TETRAHEDRONSYSTEM_IMPLEMENT_H
#define TETRAHEDRONSYSTEM_IMPLEMENT_H


#include "bvh_common.h"

extern "C" {
    void tetrahedronSystemIntegrate(float3 * o_position, float3 * i_velocity, 
                                    float dt, uint n);
}

#endif        //  #ifndef TETRAHEDRONSYSTEM_IMPLEMENT_H

