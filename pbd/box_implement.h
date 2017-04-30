#ifndef BOX_IMPLEMENT_H
#define BOX_IMPLEMENT_H

#include <cuda_runtime_api.h>

struct Aabb {
    float3 low;
    float3 high;
};

extern "C" void calculateAabb(Aabb *dst, float3 * cvs, unsigned * indices, unsigned numTriangle);

#endif        //  #ifndef BOX_IMPLEMENT_H
