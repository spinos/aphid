#ifndef SIMPLEQUEUEINTERFACE_H
#define SIMPLEQUEUEINTERFACE_H

#include "bvh_common.h"
#define SIZE_OF_SIMPLEQUEUEINTERFACE 32
struct SimpleQueueInterface {
    int * elements;
    int lock;
    uint qhead;
    uint qtail;
    uint numNodes;
    uint maxNumWorks;
    uint workDone;
    uint workBlock;
};
#endif        //  #ifndef SIMPLEQUEUEINTERFACE_H

