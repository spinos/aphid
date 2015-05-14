#ifndef SIMPLEQUEUEINTERFACE_H
#define SIMPLEQUEUEINTERFACE_H

#include "bvh_common.h"
#define SIZE_OF_SIMPLEQUEUEINTERFACE 64
struct SimpleQueueInterface {
    int * elements;
    int lock;
    int qhead;
    int qouttail;
    int qintail;
    uint numNodes;
    uint maxNumWorks;
    int workDone;
    uint workBlock;
    int padding[7];
};
#endif        //  #ifndef SIMPLEQUEUEINTERFACE_H

