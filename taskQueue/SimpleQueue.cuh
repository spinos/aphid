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
        _tbid = 0;//interface->tbid;
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
 *  return value  0 -> n-1 bid to work right now
 *               -1 wait for new work added
 *               -2 out of work
 */    
    __device__ int dequeue()
    {
        lock();
        int oldBid = _tbid;
        if(isLastInQueue()) {
            if(countNewTask()) {
                _tbid++;
            }
            else {
                oldBid = -1;
                if(isAllWorkDone())
                    oldBid = -2;
            }
        }
        else {
            _tbid++;
        }
        unlock();
        return oldBid;
        //return atomicAdd(&_tbid, 1);
    }
/*
 *  0  1  2  3 ... n-1   n      n+1  n+2
 *                 ^     ^
 *                 |     |
 *                 tbid0 qtail0
 *                                   ^                   
 *                                   |                   
 *                                   qtail1
 *                       ^           ^
 *                       |           |
 *                       tbid1       qtail0
 *
 */  
    __device__ int isLastInQueue()
    {
        return _tbid == _qinTail;
    }
    
    __device__ int countNewTask()
    {
        //lock();
        uint oldTail = _qinTail;
        _qinTail = _qoutTail;
        int c = _qinTail - oldTail;
        //unlock();
        return c;
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
    __device__ int enqueue(int2 * nodes, int low, int high)
    {
        int hasWork = 0;
        lock();
        //if(_qoutTail < _maxNumWorks) {
            hasWork = 1;
            nodes[_qoutTail].x = low;
            nodes[_qoutTail].y = high;
            _qoutTail++;
        //}
        unlock();
        return hasWork;
    }
    
    __device__ int enqueue()
    {
        lock();
        int oldTail = _qoutTail;
        _qoutTail++;
        unlock();
        return oldTail;
    }
    
    __device__ int enqueue2()
    {
        lock();
        int oldTail = _qoutTail;
        _qoutTail += 2;
        unlock();
        return oldTail;
    }
    
    __device__ uint maxNumWorks()
    {
        return _maxNumWorks;
    }
    
    __device__ void setWorkDone()
    {
        lock();
        *_workDoneCounter += 1;
        unlock();
        //atomicAdd(_workDoneCounter, 1);
    }
    
    __device__ int isAllWorkDone()
    {
        return *_workDoneCounter >= _tbid-1;
    }
    
    __device__ uint outTail()
    {
        return _qoutTail;
    }
    
    __device__ uint inTail()
    {
        return _qinTail;
    }
    
    __device__ uint tbid()
    { return _tbid; }
    
    __device__ uint workDoneCount()
    {
        return *_workDoneCounter;   
    }
};

}
#endif        //  #ifndef SIMPLEQUEUE_CUH

