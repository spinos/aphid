#ifndef CUDAGLBASE_H
#define CUDAGLBASE_H
#include "CudaBase.h"

namespace aphid {

class CudaGLBase : public CudaBase
{
public:
    CudaGLBase();
    virtual ~CudaGLBase();
    
    static void SetGLDevice();
	
private:
};

}
#endif        //  #ifndef CUDAGLBASE_H

