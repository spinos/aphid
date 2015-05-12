#ifndef SIMPLEQUEUE_CUH
#define SIMPLEQUEUE_CUH

#include <cuda_runtime_api.h>
#include "SimpleQueueInterface.h"

namespace simpleQueue {
__device__ uint *_mutex;
__device__ uint _tbid = 0;
__device__ uint _qinTail = 0;
__device__ uint _qoutTail = 0;
__device__ uint _maxNumWorks = 0;
__device__ uint *_workDoneCounter;

struct SimpleQueue {
/*
 *  n nodes                                max n nodes
 *
 *  0  1  2  3 ... n-1  n
 *  ^                   ^
 *  |                   |
 *  tbid                qtail
 *  first work          last work
 *  ^
 *  |
 *  workDoneCounter
 *  n finished works
 *
 */
     inline __device__ void init(SimpleQueueInterface * interface) 
    {
        _mutex = &interface->lock;
        _workDoneCounter = &interface->workDone;
        _tbid = interface->tbid;
        _qinTail = _qoutTail = 0;
        _maxNumWorks = interface->maxNumWorks;   
    }
    
    __device__ void lock()
    {
        while( atomicCAS( _mutex, 0, 1 ) != 0 );
    }
    
    __device__ void unlock() 
    {
        atomicExch( _mutex, 0 );
    }
/*
 *  0  1  2  3 ... n-1  n
 *  ^                   ^
 *  |                   |
 *  tbid0               qtail
 *     ^                   
 *     |                   
 *     tbid1                       
 *
 */    
    __device__ int dequeue()
    {
        return atomicAdd(&_tbid, 1);
    }
/*
 *  0  1  2  3 ... n-1   n    n+1
 *     ^                 ^
 *     |                 |
 *     tbid              qtail0
 *                            ^                   
 *                            |                   
 *                            qtail1                       
 *
 */        
    __device__ int enqueue()
    {
        return atomicAdd(&_qoutTail, 1);
    }
    
    __device__ uint maxNumWorks()
    {
        return _maxNumWorks;
    }
    
    __device__ void setWorkDone()
    {
        atomicAdd(_workDoneCounter, 1);
    }
    
    __device__ int isAllWorkDone()
    {
        return *_workDoneCounter >= _qinTail;
    }
    
    __device__ int countNewTask()
    {
        lock();
        uint oldTail = _qinTail;
        _qinTail = _qoutTail;
        int c = _qinTail - oldTail;
        unlock();
        return c;
    }
    
    __device__ uint outTail()
    {
        return _qoutTail;
    }
    
    __device__ uint workDoneCount()
    {
        return *_workDoneCounter;   
    }
};

}
#endif        //  #ifndef SIMPLEQUEUE_CUH

