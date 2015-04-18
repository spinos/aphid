#include "CudaReduction.h"
#include <CUDABuffer.h>
#include <cuReduceSum_implement.h>
CudaReduction::CudaReduction()
{
    m_obuf = new CUDABuffer;
}

CudaReduction::~CudaReduction()
{
    delete m_obuf;
}

void CudaReduction::initOnDevice()
{
    m_obuf->create(ReduceMaxBlocks * 4);
}

void CudaReduction::sumF(float & result, float * idata, unsigned m)
{
    uint threads, blocks;
    uint n = m;
	getReduceBlockThread(blocks, threads, n);
	
	// std::cout<<"n "<<n<<" blocks x threads : "<<blocks<<" x "<<threads<<" sharedmem size "<<threads * sizeof(float)<<"\n";
	
	void * d_odata = m_obuf->bufferOnDevice();
	cuReduce_F_Sum((float *)d_odata, idata, m, blocks, threads);
	
	n = blocks;	
	while(n > 1) {
	    blocks = threads = 0;
		getReduceBlockThread(blocks, threads, n);
		
		// std::cout<<"n "<<n<<" blocks x threads : "<<blocks<<" x "<<threads<<" sharedmem size "<<threads * sizeof(float)<<"\n";
	
		cuReduce_F_Sum((float *)d_odata, (float *)d_odata, n, blocks, threads);
		
		n = (n + (threads*2-1)) / (threads*2);
	}

	cudaMemcpy(&result, d_odata, 4, cudaMemcpyDeviceToHost);	
}
