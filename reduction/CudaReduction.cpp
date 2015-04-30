#include "CudaReduction.h"

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
    m_obuf->create(ReduceMaxBlocks * 32);
}

