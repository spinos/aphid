#ifndef CUDACSRMATRIX_H
#define CUDACSRMATRIX_H

#include "CSRMatrix.h"

class CUDABuffer;

class CudaCSRMatrix : public CSRMatrix
{
public:
    CudaCSRMatrix();
    virtual ~CudaCSRMatrix();
    
    void initOnDevice();
    CUDABuffer * valueBuf();
    CUDABuffer * rowPtrBuf();
    CUDABuffer * colIndBuf();
protected:

private:
    CUDABuffer * m_deviceValue;
    CUDABuffer * m_deviceRowPtr;
    CUDABuffer * m_deviceColInd;
};

#endif        //  #ifndef CUDACSRMATRIX_H

