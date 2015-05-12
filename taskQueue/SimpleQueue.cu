#include "SimpleQueue.cuh"

namespace simpleQueue {
__global__ void initSimpleQueue_kernel(SimpleQueue * q,
                                    uint tail,
                                    uint maxN)
{
    if(blockIdx.x < 1 && threadIdx.x < 1)
        q->init(tail, maxN);
}
}
