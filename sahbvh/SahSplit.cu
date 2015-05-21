#include "SahInterface.h"

#include "SahSplit.cuh"

namespace sahsplit {
void doSplitWorks(void * q, int * qelements,
                    int2 * nodes,
                    Aabb * nodeAabbs,
                    KeyValuePair * primitiveIndirections,
                    Aabb * primitiveAabbs,
                    KeyValuePair * intermediateIndirections,
                    uint numPrimitives)
{
    simpleQueue::SimpleQueue * queue = (simpleQueue::SimpleQueue *)q;
    simpleQueue::init_kernel<<< 1,32 >>>(queue, qelements);
    
    DataInterface data;
    data.nodes = nodes;
    data.nodeAabbs = nodeAabbs;
    data.primitiveIndirections = primitiveIndirections;
    data.primitiveAabbs = primitiveAabbs;
    data.intermediateIndirections = intermediateIndirections;
    
    SplitTask task;
    
    const int tpb = 256;
    dim3 block(tpb, 1, 1);
    const unsigned nblk = 1024;
    dim3 grid(nblk, 1, 1);
    
    int lpb = 1 + numPrimitives>>10;
    
    work_kernel<simpleQueue::SimpleQueue, SplitTask, DataInterface, 24, 8, 256><<<grid, block, 16320>>>(queue,
                                task,
                                data,
                                lpb,
                                31);
}
}
