#ifndef CUDABASE_H
#define CUDABASE_H

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
    
    static int MemoryUsed;
private:
    
};

#endif        //  #ifndef CUDAWORKS_H

