#include <QtCore>
#include "CudaBase.h"
#include <cuda_runtime_api.h>
CudaBase::CudaBase()
{
    
}

CudaBase::~CudaBase()
{
    
}

char CudaBase::checkCUDevice()
{
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
            qDebug() << "Cannot find CUDA device!";
        return 0;
    }
    
    if(deviceCount>0) {
            qDebug() << "Found " << deviceCount << " device(s)";
            int driverVersion = 0, runtimeVersion = 0;
            cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
            cudaDriverGetVersion(&driverVersion);
            cudaRuntimeGetVersion(&runtimeVersion);
            qDebug() << "Device name: " << deviceProp.name;
            qDebug() << "  Diver Version: " << driverVersion;
            qDebug() << "  Runtime Version: " << runtimeVersion;
            
    qDebug() << QString("  Maximum sizes of each dimension of a grid: %1 x %2 x %3")
           .arg(deviceProp.maxGridSize[0]).arg(deviceProp.maxGridSize[1]).arg(deviceProp.maxGridSize[2]);
    
    qDebug() << QString("  Maximum sizes of each dimension of a block: %1 x %2 x %3")
                 .arg(deviceProp.maxThreadsDim[0]).arg(deviceProp.maxThreadsDim[1]).arg(deviceProp.maxThreadsDim[2]);
            qDebug() << "  Maximum number of threads per block: " << deviceProp.maxThreadsPerBlock;
        return 1;               
    }
    return 0;
}
