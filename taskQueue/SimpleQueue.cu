#include "SimpleQueue.cuh"

namespace simpleQueue {
__global__ void init_kernel(SimpleQueue * queue,
                            int * elements)
{
    if(threadIdx.x < 1) {
        queue->init();
        queue->setElements(elements);
    }
}
}
