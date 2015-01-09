#ifndef CUDABASE_H
#define CUDABASE_H

class CudaBase
{
public:
    CudaBase();
    virtual ~CudaBase();
    
    static char CheckCUDevice();
    static void SetDevice();
private:
    
};

#endif        //  #ifndef CUDAWORKS_H

