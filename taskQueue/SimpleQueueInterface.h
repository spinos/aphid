#ifndef SIMPLEQUEUEINTERFACE_H
#define SIMPLEQUEUEINTERFACE_H

#include "bvh_common.h"
#define SIZE_OF_SIMPLEQUEUE 32
#define SIZE_OF_SIMPLEQUEUEINTERFACE 32
struct SimpleQueueInterface {
    int qhead;
    int qouttail;
    int qintail;
    int workDone;
    int workBlock;
    int lastBlock;
    int padding[2];
};
#endif        //  #ifndef SIMPLEQUEUEINTERFACE_H

