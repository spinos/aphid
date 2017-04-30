#ifndef BBOX_IMPLEMENT_H
#define BBOX_IMPLEMENT_H

#include <cuda_runtime_api.h>
typedef unsigned int uint;
struct Aabb {
    float3 low;
    float3 high;
};

extern "C" void calculateAabbs(Aabb *dst, float3 * cvs, unsigned * indices, unsigned numTriangle);

#endif        //  #ifndef BOX_IMPLEMENT_H
