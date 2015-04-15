#ifndef CUFEMTETRAHEDRON_IMPLEMENT_H
#define CUFEMTETRAHEDRON_IMPLEMENT_H

#include "bvh_common.h"

extern "C" {
    
void cuFemTetrahedron_resetRe(mat33 * d, uint maxInd);

}
#endif        //  #ifndef CUFEMTETRAHEDRON_IMPLEMENT_H

