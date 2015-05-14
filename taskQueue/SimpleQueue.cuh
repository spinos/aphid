#ifndef SIMPLEQUEUE_CUH
#define SIMPLEQUEUE_CUH

#include <cuda_runtime_api.h>
#include "SimpleQueueInterface.h"

namespace simpleQueue {
__device__ int * _mutex;
__device__ int * _elements;
__device__ int * _qintail;
__device__ int * _qouttail;
__device__ int * _qhead;
__device__ uint _maxNumWorks = 0;
__device__ int *_workDoneCounter;

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
    __device__ void init(SimpleQueueInterface * interface) 
    {
        if(blockIdx.x < 1 && threadIdx.x < 1) {
            _mutex = &interface->lock;
            _elements = interface->elements;
            _workDoneCounter = &interface->workDone;
            _qhead = &interface->qhead;
            _qintail = &interface->qintail;
            _qouttail = &interface->qouttail;
            _maxNumWorks = interface->maxNumWorks;
            
            *_qhead = 0;
            *_qintail = 1;
            *_qouttail = 1;
            _elements[0] = 0;
        }
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
        int oldTail = atomicAdd(_qouttail, 1);
        /*lock();
        int oldTail = _qouttail;
        _qouttail++;
        
        unlock();*/
        _elements[oldTail] = 0;
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
        int oldTail = *_qintail;
        int i = *_qhead;
        for(;i<oldTail;i++) {
            if(atomicCAS( &_elements[i], 0, 1 ) == 0) {
                *_qhead=i;
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
    
    __device__ int isEmpty()
    {
        return _maxNumWorks < 1;
    }
    
    __device__ void setWorkDone()
    {
        atomicAdd(_workDoneCounter, 1);
    }
    
    __device__ uint head()
    {
        return *_qhead;
    }
    
    __device__ uint intail()
    {
        return *_qintail;
    }
    
    __device__ int outtail()
    {
        return *_qouttail;
    }
    
    __device__ uint workDoneCount()
    {
        return *_workDoneCounter;   
    }
    
    __device__ void swapTails()
    {
        lock();
        if(*_qintail== *_workDoneCounter) {
            
                *_qhead = *_qintail;
                *_qintail = *_qouttail;
            
        }
        unlock();
    }
};

template<unsigned long StaticLimit>
struct TimeLimiter
{
    unsigned long  TimeLimiter_start;
    int _isStarted;
    __device__ __inline__ TimeLimiter() 
    {
        _isStarted = 0;
    }
    __device__ __inline__ void start()
    {
      if(threadIdx.x < 1)
        TimeLimiter_start = clock();
      _isStarted = 1;
    }
    __device__ __inline__ bool stop()
    {
        if(_isStarted)
            return (clock() - TimeLimiter_start) > StaticLimit;
        return false;
    }
};
  
}
#endif        //  #ifndef SIMPLEQUEUE_CUH

