#ifndef SIMPLEQUEUE_CUH
#define SIMPLEQUEUE_CUH

#include <cuda_runtime_api.h>
#include "SimpleQueueInterface.h"

namespace simpleQueue {
__device__ int *_mutex;
__device__ int *_elements;
__device__ uint _tbid = 0;
__device__ uint _qtail = 0;
__device__ uint _qhead = 0;
__device__ uint _maxNumWorks = 0;
__device__ uint *_workDoneCounter;

struct SimpleQueue {
/*
 *  n nodes                                max n nodes
 *
 *  0  1  2  3 ... n-1  n
 *
 *  1  0  0  0     0    0                  mutex per element
 *  ^                   ^
 *  |                   |
 *  qhead               qtail
 *  first work          end of work
 *  ^
 *  |
 *  workDoneCounter
 *  n finished works
 *
 */
     inline __device__ void init(SimpleQueueInterface * interface) 
    {
        _mutex = &interface->lock;
        _elements = interface->elements;
        _workDoneCounter = &interface->workDone;
        _qhead = interface->qhead;
        _qtail = interface->qtail;
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
 *  0  1  2  3 ... n-1   n    n+1
 *
 *                       0
 *     ^                 ^
 *     |                 |
 *     qhead             qtail0
 *                            ^                   
 *                            |                   
 *                            qtail1                       
 *
 *  tail is updated atomically
 *  set added elements 0 (unlocked)
 */     
    __device__ int enqueue()
    {
        //lock();
        int oldTail = atomicAdd(&_qtail, 1);
        //_qtail++;
        _elements[oldTail] = 0;
        //unlock();
        return oldTail;
        // return atomicAdd(&_qtail, 1);
    }
    
/*
 *  0  1  2  3 ... n-1  n
 *
 *        0
 *        ^             ^
 *        |             |
 *        qhead0        qtail
 *
 *        1      
 *           ^
 *           |                   
 *           qhead1                       
 *
 *  try to atomically lock the head element
 *  if successful, move qhead forward, so shrink the active queue
 *  return value  qhead 0 -> n-1 first work available right now
 *               -1 if qhead == qtail queue is finished or qhead is alread locked
 *               -2 if no unfinished pregress out of work
 */    
    __device__ int dequeue()
    {        
        //if(isQueueFinished()) return -1;
        int oldTail = _qtail;
        int i = _qhead;
        for(;i<oldTail;i++) {
            if(atomicCAS( &_elements[i], 0, 1 ) == 0) {
                _qhead = i;
                return i;
            }
        }
        return -1;
        /*if( atomicCAS( &_elements[_qhead], 0, 1 ) != 0 ) return -1;
        int oldHead = _qhead;
        _qhead++;
        return oldHead;*/
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
        return (*_workDoneCounter >= _qtail);
    }
    
    __device__ uint head()
    {
        return _qhead;
    }
    
    __device__ uint tail()
    {
        return _qtail;
    }
    
    __device__ uint workDoneCount()
    {
        return *_workDoneCounter;   
    }
};

}
#endif        //  #ifndef SIMPLEQUEUE_CUH

