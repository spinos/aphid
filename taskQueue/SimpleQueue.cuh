#ifndef SIMPLEQUEUE_CUH
#define SIMPLEQUEUE_CUH

#include <cuda_runtime_api.h>
#include "bvh_common.h"

namespace simpleQueue {
__device__ uint _mutex;
__device__ uint _tbid;
__device__ uint _qinTail;
__device__ uint _qoutTail;
__device__ uint _maxNumWorks;

struct SimpleQueue {
     inline __device__ void init(uint tail, uint n) 
    {
        _mutex = 0;
        _tbid = 0;
        _qinTail = tail; 
        _qoutTail = tail;
        _maxNumWorks = n;
    }
    
    __device__ void lock()
    {
        while( atomicCAS( &_mutex, 0, 1 ) != 0 );
    }
    
    __device__ void unlock() 
    {
        atomicExch( &_mutex, 0 );
    }
    
    __device__ int dequeue()
    {
        lock();
        int hasWork = -1;
        if(_tbid <= _qinTail) {
            hasWork = _tbid;
            _tbid++;
        }
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
    
    __device__ uint maxNumWorks()
    {
        return _maxNumWorks;
    }
};

__global__ void initSimpleQueue_kernel(SimpleQueue * q,
                                    uint tail,
                                    uint maxN);
}
#endif        //  #ifndef SIMPLEQUEUE_CUH

