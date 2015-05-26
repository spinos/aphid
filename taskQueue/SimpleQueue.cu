#include "SimpleQueue.cuh"

namespace simpleQueue {
__global__ void init_kernel(SimpleQueue * queue,
                            int tail,
                            int * elements)
{
    if(threadIdx.x < 1) {
        queue->init(tail);
        queue->setElements(elements);
    }
}
}
