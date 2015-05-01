#include "CudaScan.h"
#include <scan_implement.h>

CudaScan::CudaScan()
{
    m_intermediate = new CUDABuffer;
}

CudaScan::~CudaScan()
{
    delete m_intermediate;
}

void CudaScan::create(unsigned n)
{
    m_intermediate->create(n * 4);
}

unsigned CudaScan::prefixSum(CUDABuffer * obuf, CUDABuffer * ibuf, unsigned n)
{ 
	scanExclusive((uint *)obuf->bufferOnDevice(), (uint *)ibuf->bufferOnDevice(), 
					(uint *)m_intermediate->bufferOnDevice(), 
					n >> 10, 1024);
					
	unsigned a=0, b=0;
    ibuf->deviceToHost(&a, 4*(n - 1), 4);
    obuf->deviceToHost(&b, 4*(n - 1), 4);
    return a + b;
}

unsigned CudaScan::getScanBufferLength(unsigned n)
{ return iRound1024(n); }