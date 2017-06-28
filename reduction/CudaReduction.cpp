#include "CudaReduction.h"

namespace aphid {

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

void * CudaReduction::resultOnDevice()
{ return m_obuf->bufferOnDevice(); }

}

