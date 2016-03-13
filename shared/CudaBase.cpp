#include <iostream>
#include "CudaBase.h"
#include <cuda_runtime.h>

namespace aphid {

int CudaBase::MaxThreadPerBlock = 512;
int CudaBase::MaxRegisterPerBlock = 8192;
int CudaBase::MaxSharedMemoryPerBlock = 16384;
int CudaBase::WarpSize = 32;
int CudaBase::RuntimeVersion = 4000;
bool CudaBase::HasDevice = 0;
int CudaBase::MemoryUsed = 0;
std::string CudaBase::BreakInfo("unknown");

CudaBase::CudaBase()
{}

CudaBase::~CudaBase()
{}

char CudaBase::CheckCUDevice()
{
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
            std::cout << "Cannot find CUDA device!";
        return 0;
    }
    
    if(deviceCount>0) {
            std::cout << "\n CudaBase found " << deviceCount << " device(s)\n";
            int driverVersion = 0, runtimeVersion = 0;
            cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
            cudaDriverGetVersion(&driverVersion);
            cudaRuntimeGetVersion(&runtimeVersion);
            std::cout << "  device name: " << deviceProp.name<<"\n"
				<< "  diver version: " << driverVersion<<"\n"
				<< "  runtime version: " << runtimeVersion<<"\n"
				<< "  capability major/minor version number: "<<deviceProp.major<<"."<<deviceProp.minor<<"\n"
				<< "  total amount of global memory: "<<(unsigned long long)deviceProp.totalGlobalMem<<" bytes\n"
				<< "  total amount of constant memory: "<<deviceProp.totalConstMem<<"bytes\n"
				<< "  total amount of shared memory per block: "<<deviceProp.sharedMemPerBlock<<" bytes\n"
				<< "  total number of registers available per block: "<<deviceProp.regsPerBlock
				<< "\n  warp size: "<<deviceProp.warpSize
				<< "\n  maximum sizes of each dimension of a grid: "<<deviceProp.maxGridSize[0]<<" x "<<deviceProp.maxGridSize[1]<<" x "<<deviceProp.maxGridSize[2]
				<< "\n  maximum sizes of each dimension of a block: "<<deviceProp.maxThreadsDim[0]<<" x "<<deviceProp.maxThreadsDim[1]<<" x "<<deviceProp.maxThreadsDim[2]
				<< "\n  maximum number of threads per block: " << deviceProp.maxThreadsPerBlock<<"\n";
            
            MaxThreadPerBlock = deviceProp.maxThreadsPerBlock;
            MaxRegisterPerBlock = deviceProp.regsPerBlock;
            MaxSharedMemoryPerBlock = deviceProp.sharedMemPerBlock;
			WarpSize = deviceProp.warpSize;
			RuntimeVersion = runtimeVersion;
        return 1;               
    }
    return 0;
}

void CudaBase::SetDevice()
{
    if(!CheckCUDevice()) return;
	cudaSetDevice(0);
	HasDevice = 1;
}

int CudaBase::LimitNThreadPerBlock(int regPT, int memPT)
{
	int tpb = MaxThreadPerBlock;
    const int byReg = MaxRegisterPerBlock / regPT;
    const int byMem = MaxSharedMemoryPerBlock / memPT;
    if(byReg < tpb) tpb = byReg;
    if(byMem < tpb) tpb = byMem;
	int nwarp = tpb / WarpSize;
	if(nwarp < 1) return WarpSize>>1;
    return nwarp*WarpSize;
}

void CudaBase::CheckCudaError(cudaError_t err, const char * info)
{
	// cudaError_t cudaResult;
    // cudaResult = cudaGetLastError();
    if (err != cudaSuccess) {
		std::cout<<"cuda last breaks "<<BreakInfo
        <<" exit due to error '"<<cudaGetErrorString(err)<<"' when "<<info<<"\n";
		exit(1);
        // Do whatever you want here
        // I normally create a std::string msg with a description of where I am
        // and append cudaGetErrorString(cudaResult)
    }
	BreakInfo = std::string(info);
}

void CudaBase::CheckCudaError(const char * info)
{ CheckCudaError(cudaGetLastError(), info); }

cudaError_t CudaBase::Synchronize()
{ return cudaDeviceSynchronize(); }

}
//:~