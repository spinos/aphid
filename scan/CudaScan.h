#ifndef CUDASCANE_H
#define CUDASCAN_H
#include <CUDABuffer.h>

class CudaScan {
public:
    CudaScan();
    virtual ~CudaScan();
    
    void create(unsigned n);
	unsigned prefixSum(CUDABuffer * obuf, CUDABuffer * ibuf, unsigned n);
	
	static unsigned getScanBufferLength(unsigned n);
protected:

private:
    CUDABuffer * m_intermediate;
};
#endif        //  #ifndef CUDASCAN_H

