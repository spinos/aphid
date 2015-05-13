#include <cuda_runtime_api.h>

#include <CUDABuffer.h>
#include <BaseBuffer.h>
#include <QuickSort.h>
#include <CudaDbgLog.h>
#include <iostream>
#include <SimpleQueueInterface.h>

CudaDbgLog qslog("qsort.txt");

cudaEvent_t start_event, stop_event;

void makeRadomUints(unsigned * a, 
                    unsigned n, unsigned keybits)
{
    std::cout<<" generating "<<n<<" random uint\n";
    int keyshiftmask = 0;
    if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
    int keymask = 0xffff;
    if (keybits < 16) keymask = (1 << keybits) - 1;

    srand(95123);
    for(unsigned int i=0; i < n; ++i) { 
        a[i] = ((rand() & keyshiftmask)<<16) | (rand() & keymask); 
    }
}

bool checkSortResult(unsigned * a, 
                    unsigned n)
{
    unsigned b = a[0];
    for(unsigned i=1; i<n;i++) {
        if(a[i]<b) {
            std::cout<<" unsorted element["<<i<<"] "<<a[i]<<" < "<<b<<" !\n";
            return false;
        }
        b = a[i];
    }
    return true;
}

extern "C" {
void cu_testQuickSort(void * q,
                    unsigned * idata,
                    unsigned * nodes, 
                    SimpleQueueInterface * qi,
                    unsigned maxNumParallelNodes,
                    unsigned * checkMaxN,
                    unsigned * workBlocks,
                    unsigned * loopbuf);
}

int main(int argc, char **argv)
{
    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    int devID = 0;
    
    cudaGetDeviceProperties(&deviceProp, devID);

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n", 
		deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = (deviceProp.major * 0x10 + deviceProp.minor);
    if(version < 0x11) 
    {
        printf("cudacg: requires a minimum CUDA compute 1.1 capability\n");
        printf("PASSED");
        cudaThreadExit();
        exit(1);
    }
        
    printf("start task queue.\n");
    
    unsigned n = 1<<16;
    BaseBuffer hdata;
    hdata.create(n*4);
    
    unsigned * hostData = (unsigned *)hdata.data();
    makeRadomUints(hostData, n, 30);
    
    CUDABuffer ddata;
    ddata.create(n*4);
    ddata.hostToDevice(hostData);
    unsigned * deviceData = (unsigned *)ddata.bufferOnDevice();
    
// max 2^16 nodes first int is n levels
    unsigned maxNumNodes = 1<<16;
    std::cout<<" max n sorting nodes "<<maxNumNodes<<"\n";
    
    CUDABuffer nodesBuf;
    nodesBuf.create((maxNumNodes * 2)*4);
    unsigned * nodes = (unsigned *)nodesBuf.bufferOnDevice();
    
// create first node
    unsigned nodeCount = 1;
    unsigned rootRange[2];
    rootRange[0] = 0;
    rootRange[1] = n - 1;
    nodesBuf.hostToDevice(&rootRange, 8);
    
    CUDABuffer elementsbuf;
    elementsbuf.create(maxNumNodes * 4);
    
    SimpleQueueInterface qi;
    qi.elements = (int *)elementsbuf.bufferOnDevice();
    qi.qhead = 0;
    qi.qtail = 0;
    qi.numNodes = 1;
    qi.maxNumWorks = maxNumNodes;
    qi.lock = 0;
    qi.workDone = 0;
    
    CUDABuffer dqi;
    dqi.create(SIZE_OF_SIMPLEQUEUEINTERFACE);
    dqi.hostToDevice(&qi, SIZE_OF_SIMPLEQUEUEINTERFACE);
    
    CUDABuffer qbuf;
    qbuf.create(8);

    CUDABuffer maxnbuf;
    maxnbuf.create(4);
    
    unsigned numParallel = 1024;
    
    CUDABuffer blkbuf;
    blkbuf.create(maxNumNodes * 4);
    CUDABuffer loopbuf;
    loopbuf.create(numParallel * 4);
    
    unsigned nine = 999;
    maxnbuf.hostToDevice(&nine, 4);
    //QuickSort1::Sort<unsigned>(hostD, 0, n-1);
    
    cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
	cudaEventRecord(start_event, 0);
    
	std::cout<<" launch cu kernel\n";
    cu_testQuickSort(qbuf.bufferOnDevice(),
                                deviceData, 
                                nodes, 
                                (SimpleQueueInterface *)dqi.bufferOnDevice(),
                                numParallel,
                                (unsigned *)maxnbuf.bufferOnDevice(),
                                (unsigned *)blkbuf.bufferOnDevice(),
                                (unsigned *)loopbuf.bufferOnDevice());
    
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    float met;
	cudaEventElapsedTime(&met, start_event, stop_event);
	std::cout<<" test "<<n<<" took "<<met<<" milliseconds\n";
		
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    
    unsigned mnw = 0;
    maxnbuf.deviceToHost(&mnw, 4);
    std::cout<<" check q max n works "<<mnw<<"\n";
    
    dqi.deviceToHost(&qi, SIZE_OF_SIMPLEQUEUEINTERFACE);
    std::cout<<" last work done by block "<<qi.workBlock<<"\n";
    std::cout<<" n work done "<<qi.workDone<<"\n";
    std::cout<<" q tail "<<qi.qtail<<"\n";
    
    qslog.writeInt2(&nodesBuf, qi.workDone, "sort_node", CudaDbgLog::FOnce);
    qslog.writeUInt(&ddata, n, "result", CudaDbgLog::FOnce);
    qslog.writeUInt(&blkbuf, qi.workDone, "work_blocks", CudaDbgLog::FOnce);
    qslog.writeUInt(&loopbuf, numParallel, "loop_blocks", CudaDbgLog::FOnce);
    
    ddata.deviceToHost(hostData);
    if(checkSortResult(hostData, n)) std::cout<<" gpu sorted passed.\n";
    printf("done.\n");
    exit(0);
}
