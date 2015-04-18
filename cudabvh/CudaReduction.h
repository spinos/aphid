#ifndef CUDAREDUCTION_H
#define CUDAREDUCTION_H

class CUDABuffer;
class CudaReduction {
public:
    CudaReduction();
    virtual ~CudaReduction();
    
    void initOnDevice();
    void sumF(float & result, float * idata, unsigned m);
protected:

private:
    CUDABuffer * m_obuf;
};
#endif        //  #ifndef CUDAREDUCTION_H

