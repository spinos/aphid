#ifndef CUDABASE_H
#define CUDABASE_H
#include <cuda_runtime_api.h>
#include <string>
class CudaBase
{
public:
    CudaBase();
    virtual ~CudaBase();
    
    static char CheckCUDevice();
    static void SetDevice();
    static void SetGLDevice();
    
    static int MaxThreadPerBlock;
    static int MaxRegisterPerBlock;
    static int MaxSharedMemoryPerBlock;
	static int WarpSize;
	static int RuntimeVersion;
	static bool HasDevice;
    static int LimitNThreadPerBlock(int regPT, int memPT);
	static void CheckCudaError(cudaError_t err, const char * info);
    static void CheckCudaError(const char * info);
    
    static int MemoryUsed;
	static std::string BreakInfo;
private:
};

#endif        //  #ifndef CUDAWORKS_H

