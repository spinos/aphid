#include <iostream>
#include <sstream>
#include <gl_heads.h>
#include "CudaBase.h"
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
CudaBase::CudaBase()
{
    
}

CudaBase::~CudaBase()
{
    
}

char CudaBase::CheckCUDevice()
{
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
            std::cout << "Cannot find CUDA device!";
        return 0;
    }
    
    if(deviceCount>0) {
            std::cout << "Found " << deviceCount << " device(s)\n";
            int driverVersion = 0, runtimeVersion = 0;
            cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
            cudaDriverGetVersion(&driverVersion);
            cudaRuntimeGetVersion(&runtimeVersion);
            std::cout << "  Device name: " << deviceProp.name<<"\n";
            std::cout << "  Diver Version: " << driverVersion<<"\n";
            std::cout << "  Runtime Version: " << runtimeVersion<<"\n";
            std::cout << "  Capability Major/Minor version number: "<<deviceProp.major<<"."<<deviceProp.minor<<"\n";
            std::cout << "  Total amount of global memory: "<<(unsigned long long)deviceProp.totalGlobalMem<<" bytes\n";
            std::cout << "  Total amount of constant memory: "<<deviceProp.totalConstMem<<"bytes\n"; 
            std::cout << "  Total amount of shared memory per block: "<<deviceProp.sharedMemPerBlock<<" bytes\n";
        
            std::stringstream sst;
            sst<<"  Maximum sizes of each dimension of a grid: "<<deviceProp.maxGridSize[0]<<" x "<<deviceProp.maxGridSize[1]<<" x "<<deviceProp.maxGridSize[2];
            std::cout<<sst.str()<<"\n";
            sst.str("");
            sst<<"  Maximum sizes of each dimension of a block: "<<deviceProp.maxThreadsDim[0]<<" x "<<deviceProp.maxThreadsDim[1]<<" x "<<deviceProp.maxThreadsDim[2];
            std::cout<<sst.str()<<"\n";
            std::cout << "  Maximum number of threads per block: " << deviceProp.maxThreadsPerBlock<<"\n";
        return 1;               
    }
    return 0;
}

void CudaBase::SetDevice()
{
	cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());
}
