#include "CudaCSRMatrix.h"
#include <CUDABuffer.h>

CudaCSRMatrix::CudaCSRMatrix() 
{
    m_deviceValue = new CUDABuffer;
    m_deviceRowPtr = new CUDABuffer;
    m_deviceColInd = new CUDABuffer;
}

CudaCSRMatrix::~CudaCSRMatrix() 
{
    delete m_deviceValue;
    delete m_deviceRowPtr;
    delete m_deviceColInd;
}

CUDABuffer * CudaCSRMatrix::valueBuf()
{ return m_deviceValue; }

CUDABuffer * CudaCSRMatrix::rowPtrBuf()
{ return m_deviceRowPtr; }

CUDABuffer * CudaCSRMatrix::colIndBuf()
{ return m_deviceColInd; }

void CudaCSRMatrix::initOnDevice()
{
    m_deviceValue->create(numNonZero() * valueType());
    m_deviceRowPtr->create((dimension() + 1) * 4);
    m_deviceColInd->create(numNonZero() * 4);
    
    m_deviceRowPtr->hostToDevice(rowPtr());
    m_deviceColInd->hostToDevice(colInd());
}

