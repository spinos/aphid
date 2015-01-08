#ifndef CREATEBVH_IMPLEMENT_H
#define CREATEBVH_IMPLEMENT_H

#include "bvh_common.h"

extern "C" void calculateAabbs(Aabb *dst, float3 * cvs, unsigned * indices, unsigned numTriangle);

#endif        //  #ifndef CREATEBVH_IMPLEMENT_H

