#ifndef CUDABASE_H
#define CUDABASE_H

class CudaBase
{
public:
    CudaBase();
    virtual ~CudaBase();
    
    static char CheckCUDevice();
    static void SetDevice();
    
    static int MaxThreadPerBlock;
    static int MaxRegisterPerBlock;
    static int MaxSharedMemoryPerBlock;
    static int LimitNThreadPerBlock(int regPT, int memPT);
private:
    
};

#endif        //  #ifndef CUDAWORKS_H

