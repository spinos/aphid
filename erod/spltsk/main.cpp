#include <radixsort_implement.h>
#include <cuReduceSum_implement.h>
#include <CudaReduction.h>
#include <CUDABuffer.h>
#include <math/BaseBuffer.h>
#include <CudaScan.h>
#include "SahInterface.h"

using namespace aphid;

cudaEvent_t start_event, stop_event;

void testSplitTask()
{
	std::cout<<"\n generat primitive box\n";
	unsigned m = (1<<12);
	BaseBuffer hb;
	hb.create(m*32);
	
	Aabb * h = (Aabb *)hb.data();
	float x, y, z;
	unsigned i;
	for(i=0; i< m; i++) {
	    x = 66.f * ((float)(rand() & 255))/256.f - 22.f;
	    y = 77.f * ((float)(rand() & 255))/256.f - 33.f;
	    z = 88.f * ((float)(rand() & 255))/256.f - 55.f;
	    h[i].low.x = x - 1.f;
	    h[i].low.y = y - 1.f;
	    h[i].low.z = z - 1.f;
	    h[i].high.x = x + 1.f;
	    h[i].high.y = y + 1.f;
	    h[i].high.z = z + 1.f;
	}
	
	printf(" copy primitvie box\n");
	CUDABuffer primitiveAabb;
	primitiveAabb.create(m*32);
	primitiveAabb.hostToDevice(hb.data());
	
	CudaReduction reducer;
	reducer.initOnDevice();
	
	printf(" start reduce box\n");
	
	cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
    
    float bounding[6];

    cudaEventRecord(start_event, 0);
        
    reducer.minMaxBox<Aabb, float3>((Aabb *)&bounding, (float3 *)primitiveAabb.bufferOnDevice(), m<<1);
    
    std::cout<<" bounding ("<<bounding[0]<<","<<bounding[1]<<","<<bounding[2]<<"),("
            <<bounding[3]<<","<<bounding[4]<<","<<bounding[5]<<")\n";
    
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    float met;
    cudaEventElapsedTime(&met, start_event, stop_event);
    std::cout<<" reduce "<<(m)<<" boxes took "<<met<<" milliseconds\n";
    
    CUDABuffer internalNodeChildIndices;
    internalNodeChildIndices.create((m-1)*8);
    CUDABuffer internalNodeAabbs;
    internalNodeAabbs.create((m-1)*32);
    CUDABuffer internalNodeParentIndices;
    internalNodeParentIndices.create((m-1)*4);
    CUDABuffer distanceInternalNodeFromRoot;
    distanceInternalNodeFromRoot.create((m-1)*4);
    CUDABuffer primitiveHash;
    primitiveHash.create(m*8);
    CUDABuffer primitiveHashIntermediate;
    primitiveHashIntermediate.create(m*8);
    
    sahdecompress::initHash((KeyValuePair *)primitiveHash.bufferOnDevice(),
                             m);
    
    int rr[2];
    rr[0] = 0;
    rr[1] = m-1;
    
    internalNodeChildIndices.hostToDevice(rr, 8);
	internalNodeAabbs.hostToDevice(bounding, 24);
    int zero = 0;
    internalNodeParentIndices.hostToDevice(&zero, 4);
    distanceInternalNodeFromRoot.hostToDevice(&zero, 4);
    
    CUDABuffer queueAndElement;
    queueAndElement.create(SIZE_OF_SIMPLEQUEUE + m * 4);
    
    cudaEventRecord(start_event, 0);
    
    int n = sahsplit::doSplitWorks(queueAndElement.bufferOnDevice(),
        (int *)queueAndElement.bufferOnDeviceAt(SIZE_OF_SIMPLEQUEUE),
        (int2 *)internalNodeChildIndices.bufferOnDevice(),
	    (Aabb *)internalNodeAabbs.bufferOnDevice(),
        (int *)internalNodeParentIndices.bufferOnDevice(),
        (int *)distanceInternalNodeFromRoot.bufferOnDevice(),
	    (KeyValuePair *)primitiveHash.bufferOnDevice(),
        (Aabb *)primitiveAabb.bufferOnDevice(),
        (KeyValuePair *)primitiveHashIntermediate.bufferOnDevice(),
        m,
        1);
    std::cout<<" n node "<<n<<"\n";
    
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&met, start_event, stop_event);
    std::cout<<" split "<<(m)<<" boxes took "<<met<<" milliseconds\n";
    
	cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
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
        
    printf("test sah split task\n");
    testSplitTask();
	printf("done.\n");
    exit(0);
}
