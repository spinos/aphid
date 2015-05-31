#ifndef TETRAHEDRONSYSTEMINTERFACE_H
#define TETRAHEDRONSYSTEMINTERFACE_H

#include "bvh_common.h"
#define TETRAHEDRONSYSTEM_VICINITY_LENGTH 32

namespace tetrasys {
    
void writeVicinity(int * vicinities,
                    int * indices,
                    int * offsets,
                    uint n);
}
#endif        //  #ifndef TETRAHEDRONSYSTEMINTERFACE_H

