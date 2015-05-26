#include "SahSplit.cuh"

namespace sahsplit {
int doSplitWorks(void * q, int * qelements,
                    int2 * nodes,
                    Aabb * nodeAabbs,
                    int * nodeParents,
                    int * nodeLevels,
                    KeyValuePair * primitiveIndirections,
                    Aabb * primitiveAabbs,
                    KeyValuePair * intermediateIndirections,
                    uint numPrimitives,
                    int initialNumNodes)
{
    simpleQueue::SimpleQueue * queue = (simpleQueue::SimpleQueue *)q;
    simpleQueue::init_kernel<<< 1,32 >>>(queue, initialNumNodes, qelements);
    
    DataInterface data;
    data.nodes = nodes;
    data.nodeAabbs = nodeAabbs;
    data.nodeParents = nodeParents;
    data.nodeLevels = nodeLevels;
    data.primitiveIndirections = primitiveIndirections;
    data.primitiveAabbs = primitiveAabbs;
    data.intermediateIndirections = intermediateIndirections;
    
    SplitTask task;
    
    const int tpb = 256;
    dim3 block(tpb, 1, 1);
    const unsigned nblk = iRound1024(numPrimitives);
    dim3 grid(nblk, 1, 1);
    
    int lpb = 1 + nblk>>10;
    
    work_kernel<simpleQueue::SimpleQueue, SplitTask, DataInterface, 23, 8, 256><<<grid, block, 16320>>>(queue,
                                task,
                                data,
                                lpb,
                                numPrimitives-1);
                                
    simpleQueue::SimpleQueue result;
    cudaError_t err = cudaMemcpy(&result, queue, SIZE_OF_SIMPLEQUEUE, cudaMemcpyDeviceToHost); 
    if (err != cudaSuccess) {
        printf(" cu error %s when retrieving task queue result\n", cudaGetErrorString(err));
    }
    /*
    printf("\nstart tail %i\n", initialNumNodes);
    printf("q head %i\n", result._qhead);
    printf("q in tail %i\n", result._qintail);
    printf("q out tail %i\n", result._qouttail);
    printf("q work done %i\n", result._workDoneCounter);
    */
    return result._qouttail;
}
}
