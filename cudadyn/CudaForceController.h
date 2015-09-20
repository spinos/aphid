#ifndef CUDAFORCECONTROLLER_H
#define CUDAFORCECONTROLLER_H
class CUDABuffer;
class CudaForceController {
public:
    CudaForceController();
    virtual ~CudaForceController();
    
    void setNumNodes(unsigned x);
    void setGravity(float x, float y, float z);
    void setMassBuf(CUDABuffer * x);
    void setImpulseBuf(CUDABuffer * x);
    
    void updateGravity(float dt);
    
protected:

private:
    CUDABuffer * m_mass;
    CUDABuffer * m_impulse;
    float m_gravity[3];
    unsigned m_numNodes;
};
#endif        //  #ifndef CUDAFORCECONTROLLER_H

