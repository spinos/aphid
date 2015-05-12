/*
 *  sort node count and range
 *
 *  1    0,13
 *  2    0,6  7,13
 *  4    0,3  4,6  7,10  11,13
 *  8    0,1  2,3  4,5   5,6   7,8  9,10  11,12  13,13
 *
 *  each node consumes one spawns two or one or zero
 *  level n needs max 2^n nodes
 *  each level has node count and range updated by atomic
 *  
 *  taskIn  1  0,13 |
 *  taskOut 3  0,13 | 0,6  7,13 
 *
 *  taskIn  3  0,13   0,6  7,13 |
 *  taskOut 7  0,13   0,6  7,13 | 0,3   4,6  7,10  11,13
 *
 *  taskIn  7  0,13   0,6  7,13   0,3   4,6  7,10  11,13 |
 *  taskOut 15 0,13   0,6  7,13   0,3   4,6  7,10  11,13 | 0,1  2,3  4,5   5,6   7,8  9,10  11,12  13,13
 *
 *  queue has   tbit where the first task is available
 *              qtail where the last task is available
 *  
 *  0  1  2  3 ... n-1
 *  ^              ^
 *  |              |
 *  tbid           qtail
 *
 *  dequeue() return task[tbid] tbid++
 *  only taskIn can be dequeued
 *  enqueue() set task[qtail] qtail++
 *  only taskOut can be enqueued
 *
 *  when taskIn.tbid > taskOut.qtail means no more work is available
 *  no more work is available doesn't mean the job is done
 *  still could be more task added, need workDoneCounter
 *  once workDoneCounter > taskOut.qtail, there is nothing left to do
 *  intially task out is a copy of task in
 *  new tasks will be added to end
 *  each time a task is done first enqueue() of needed, 
 *  then check if work is done, if not then dequeue()
 *  
 */
 
#include "quickSort.cuh"
#include "bvh_common.h"

extern "C" {
void cu_testQuickSort(void * q,
                    uint * idata,
                    uint * nodes, 
                    uint numNodes,
                    uint maxNumNodes,
                    uint maxNumParallelNodes,
                    uint * checkMaxN)
{
    printf("cu test quicksort\n");
    printf("init q max n works %i\n", maxNumNodes);
    simpleQueue::initSimpleQueue_kernel<<<512, 512>>>((simpleQueue::SimpleQueue *)q,
        numNodes -1,
        maxNumNodes);
    
    cudaDeviceSynchronize();
    
    // SimpleQueue q(lock, tail, nodeRanges);
    const int tpb = 256;
    dim3 block(tpb, 1, 1);
// one warp per parallel node
    const unsigned nblk = maxNumParallelNodes>>5;
    dim3 grid(nblk, 1, 1);
    
    quickSort_checkQ_kernel<<<grid, block>>>(checkMaxN, (simpleQueue::SimpleQueue *)q);
}

}

