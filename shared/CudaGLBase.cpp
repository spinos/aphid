#include <iostream>
#include <sstream>
#include <gl_heads.h>
#include "CudaGLBase.h"
#include <cuda_gl_interop.h>

namespace aphid {

CudaGLBase::CudaGLBase()
{}

CudaGLBase::~CudaGLBase()
{}

void CudaGLBase::SetGLDevice()
{
    if(!CudaBase::CheckCUDevice()) return;
	std::cout<<" cuda set GL device\n";
	cudaGLSetGLDevice(0);
	HasDevice = 1;
}

}
//:~