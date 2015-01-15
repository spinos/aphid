#ifndef PARTICLESYSTEM_IMPLEMENT_H
#define PARTICLESYSTEM_IMPLEMENT_H

#include "bvh_common.h"

extern "C" void particleSystemSimpleGravityForce(float3 * o_force, uint n);
extern "C" void particleSystemIntegrate(float3 * o_position, float3 * o_velocity, 
                                    float3 * force, float dt, uint n);
#endif        //  #ifndef PARTICLESYSTEM_IMPLEMENT_H

