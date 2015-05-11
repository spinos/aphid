#include <cuda_runtime_api.h>

#include <CUDABuffer.h>
#include <BaseBuffer.h>
#include <QuickSort.h>
#include <iostream>

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
void cu_testQuickSort(unsigned * idata,
                    unsigned * nNodes, unsigned * nodeRanges,
                    unsigned maxNumNodes);
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
    
    unsigned n = 1<<22;
    BaseBuffer hdata;
    hdata.create(n*4);
    
    unsigned * hostData = (unsigned *)hdata.data();
    makeRadomUints(hostData, n, 16);
    
    CUDABuffer ddata;
    ddata.create(n*4);
    ddata.hostToDevice(hostData);
    unsigned * deviceData = (unsigned *)ddata.bufferOnDevice();
    
// max 2^16 nodes first int is n nodes
    unsigned maxNumNodes = 1<<16;
    std::cout<<" max n sorting nodes "<<maxNumNodes<<"\n";
    
    CUDABuffer nodesBuf;
    nodesBuf.create((maxNumNodes + 1)*4);
    unsigned * nodes = (unsigned *)nodesBuf.bufferOnDevice();
    
// create first node
    unsigned nodeCount = 1;
    nodesBuf.hostToDevice(&nodeCount, 4);
    unsigned rootRange[2];
    rootRange[0] = 0;
    rootRange[1] = n - 1;
    nodesBuf.hostToDevice(&rootRange, 4, 8);
    
    cu_testQuickSort(deviceData, nodes, &nodes[1], maxNumNodes);
    
    //QuickSort1::Sort<unsigned>(hostD, 0, n-1);
    
    /*cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
	cudaEventRecord(start_event, 0);
    
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    float met;
	cudaEventElapsedTime(&met, start_event, stop_event);
	// std::cout<<" quicksort "<<n<<" ints took "<<met<<" milliseconds\n";
		
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);*/
    
    // if(checkSortResult(hostData, n)) std::cout<<" cpu sorted passed.\n";
    printf("done.\n");
    exit(0);
}
