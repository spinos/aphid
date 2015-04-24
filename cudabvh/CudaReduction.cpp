#include "CudaReduction.h"
#include <CUDABuffer.h>
#include <cuReduceSum_implement.h>
#include <reduceRange_implement.h>
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
    m_obuf->create(ReduceMaxBlocks * 16);
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

void CudaReduction::maxI(int & result, int * idata, unsigned m)
{
	uint threads, blocks;
    uint n = m;
	getReduceBlockThread(blocks, threads, n);
	
	void * d_odata = m_obuf->bufferOnDevice();
	bvhReduceFindMax((int *)d_odata, idata, m, blocks, threads);
	
	n = blocks;	
	while(n > 1) {
	    blocks = threads = 0;
		getReduceBlockThread(blocks, threads, n);
		
		bvhReduceFindMax((int *)d_odata, (int *)d_odata, n, blocks, threads);
		
		n = (n + (threads*2-1)) / (threads*2);
	}

	cudaMemcpy(&result, d_odata, 4, cudaMemcpyDeviceToHost);
}

void CudaReduction::maxF(float & result, float * idata, unsigned m)
{
    uint threads, blocks;
    uint n = m;
	getReduceBlockThread(blocks, threads, n);
	
	void * d_odata = m_obuf->bufferOnDevice();
	cuReduce_F_Max((float *)d_odata, idata, m, blocks, threads);
	
	n = blocks;	
	while(n > 1) {
	    blocks = threads = 0;
		getReduceBlockThread(blocks, threads, n);
		
		cuReduce_F_Max((float *)d_odata, (float *)d_odata, n, blocks, threads);
		
		n = (n + (threads*2-1)) / (threads*2);
	}

	cudaMemcpy(&result, d_odata, 4, cudaMemcpyDeviceToHost);
}

void CudaReduction::minF(float & result, float * idata, unsigned m)
{
    uint threads, blocks;
    uint n = m;
	getReduceBlockThread(blocks, threads, n);
	
	void * d_odata = m_obuf->bufferOnDevice();
	cuReduce_F_Min((float *)d_odata, idata, m, blocks, threads);
	
	n = blocks;	
	while(n > 1) {
	    blocks = threads = 0;
		getReduceBlockThread(blocks, threads, n);
		
		cuReduce_F_Min((float *)d_odata, (float *)d_odata, n, blocks, threads);
		
		n = (n + (threads*2-1)) / (threads*2);
	}

	cudaMemcpy(&result, d_odata, 4, cudaMemcpyDeviceToHost);
}

void CudaReduction::minMaxF(float * result, float * idata, unsigned m)
{
    uint threads, blocks;
    uint n = m;
	getReduceBlockThread(blocks, threads, n);
	
	void * d_odata = m_obuf->bufferOnDevice();
	cuReduce_F_MinMax1((float2 *)d_odata, idata, m, blocks, threads);
	
	n = blocks;	
	while(n > 1) {
	    blocks = threads = 0;
		getReduceBlockThread(blocks, threads, n);
		
		cuReduce_F_MinMax2((float2 *)d_odata, (float2 *)d_odata, n, blocks, threads);
		
		n = (n + (threads*2-1)) / (threads*2);
	}

	cudaMemcpy(result, d_odata, 8, cudaMemcpyDeviceToHost);
}

void CudaReduction::minBox(float * result, Aabb * idata, unsigned m)
{
    uint threads, blocks;
    uint n = m;
	getReduceBlockThread(blocks, threads, n);
	
	void * d_odata = m_obuf->bufferOnDevice();
	cuReduce_Box_Min1((float3 *)d_odata, idata, m, blocks, threads);
	
	n = blocks;	
	while(n > 1) {
	    blocks = threads = 0;
		getReduceBlockThread(blocks, threads, n);
		
		cuReduce_Pnt_Min2((float3 *)d_odata, (float3 *)d_odata, n, blocks, threads);
		
		n = (n + (threads*2-1)) / (threads*2);
	}

	cudaMemcpy(result, d_odata, 12, cudaMemcpyDeviceToHost);
}

void CudaReduction::maxBox(float * result, Aabb * idata, unsigned m)
{
    uint threads, blocks;
    uint n = m;
	getReduceBlockThread(blocks, threads, n);
	
	void * d_odata = m_obuf->bufferOnDevice();
	cuReduce_Box_Max1((float3 *)d_odata, idata, m, blocks, threads);
	
	n = blocks;	
	while(n > 1) {
	    blocks = threads = 0;
		getReduceBlockThread(blocks, threads, n);
		
		cuReduce_Pnt_Max2((float3 *)d_odata, (float3 *)d_odata, n, blocks, threads);
		
		n = (n + (threads*2-1)) / (threads*2);
	}

	cudaMemcpy(result, d_odata, 12, cudaMemcpyDeviceToHost);
}

void CudaReduction::minPnt(float * result, float3 * idata, unsigned m)
{
        uint threads, blocks;
    uint n = m;
	getReduceBlockThread(blocks, threads, n);
	
	void * d_odata = m_obuf->bufferOnDevice();
	cuReduce_Pnt_Min1((float3 *)d_odata, idata, m, blocks, threads);
	
	n = blocks;	
	while(n > 1) {
	    blocks = threads = 0;
		getReduceBlockThread(blocks, threads, n);
		
		cuReduce_Pnt_Min2((float3 *)d_odata, (float3 *)d_odata, n, blocks, threads);
		
		n = (n + (threads*2-1)) / (threads*2);
	}

	cudaMemcpy(result, d_odata, 12, cudaMemcpyDeviceToHost);
}
	
void CudaReduction::maxPnt(float * result, float3 * idata, unsigned m)
{
    uint threads, blocks;
    uint n = m;
	getReduceBlockThread(blocks, threads, n);
	
	void * d_odata = m_obuf->bufferOnDevice();
	cuReduce_Pnt_Max1((float3 *)d_odata, idata, m, blocks, threads);
	
	n = blocks;	
	while(n > 1) {
	    blocks = threads = 0;
		getReduceBlockThread(blocks, threads, n);
		
		cuReduce_Pnt_Max2((float3 *)d_odata, (float3 *)d_odata, n, blocks, threads);
		
		n = (n + (threads*2-1)) / (threads*2);
	}

	cudaMemcpy(result, d_odata, 12, cudaMemcpyDeviceToHost);
}

void CudaReduction::minX(float * result, Aabb * idata, unsigned m)
{
    uint threads, blocks;
    uint n = m;
	getReduceBlockThread(blocks, threads, n);
	
	void * d_odata = m_obuf->bufferOnDevice();
	cuReduce_Box_MinX((float *)d_odata, idata, m, blocks, threads);
	
	n = blocks;	
	while(n > 1) {
	    blocks = threads = 0;
		getReduceBlockThread(blocks, threads, n);
		
		cuReduce_F_Min((float *)d_odata, (float *)d_odata, n, blocks, threads);
		
		n = (n + (threads*2-1)) / (threads*2);
	}

	cudaMemcpy(result, d_odata, 4, cudaMemcpyDeviceToHost);
}

void CudaReduction::minPntX(float * result, float3 * idata, unsigned m)
{
    uint threads, blocks;
    uint n = m;
	getReduceBlockThread(blocks, threads, n);
	
	void * d_odata = m_obuf->bufferOnDevice();
	cuReduce_Pnt_MinX((float *)d_odata, idata, m, blocks, threads);
	
	n = blocks;	
	while(n > 1) {
	    blocks = threads = 0;
		getReduceBlockThread(blocks, threads, n);
		
		cuReduce_F_Min((float *)d_odata, (float *)d_odata, n, blocks, threads);
		
		n = (n + (threads*2-1)) / (threads*2);
	}

	cudaMemcpy(result, d_odata, 4, cudaMemcpyDeviceToHost);
}
