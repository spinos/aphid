#ifndef CUFEMTETRAHEDRON_IMPLEMENT_H
#define CUFEMTETRAHEDRON_IMPLEMENT_H

#include "bvh_common.h"

extern "C" {
    
void cuFemTetrahedron_resetRe(mat33 * d, uint maxInd);

void cuFemTetrahedron_calculateRe(mat33 * dst, 
                                    float3 * pos, 
                                    float3 * pos0,
                                    uint4 * indices,
                                    uint maxInd);
}
#endif        //  #ifndef CUFEMTETRAHEDRON_IMPLEMENT_H

