#include "CudaCSRMatrix.h"
#include <CUDABuffer.h>
#include <BaseLog.h>
#include <boost/format.hpp>
#include <Matrix33F.h>
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

void * CudaCSRMatrix::deviceValue()
{ return m_deviceValue->bufferOnDevice(); }

void * CudaCSRMatrix::deviceRowPtr()
{ return m_deviceRowPtr->bufferOnDevice(); }

void * CudaCSRMatrix::deviceColInd()
{ return m_deviceColInd->bufferOnDevice(); }

void CudaCSRMatrix::initOnDevice()
{
    m_deviceValue->create(numNonZero() * valueType());
    m_deviceRowPtr->create((dimension() + 1) * 4);
    m_deviceColInd->create(numNonZero() * 4);
    
    m_deviceRowPtr->hostToDevice(rowPtr());
    m_deviceColInd->hostToDevice(colInd());
}

void CudaCSRMatrix::print(BaseLog * lg)
{
    valueBuf()->deviceToHost(value());
    Matrix33F * hostK = (Matrix33F *)value();
    unsigned i;
    for(i=0; i<numNonZero(); i++) {
        lg->write(boost::str(boost::format("K[%1%] %2%\n") % i % hostK[i].str()));
    }
}
