#ifndef CUDAFORCECONTROLLER_H
#define CUDAFORCECONTROLLER_H
#include <Vector3F.h>
class CUDABuffer;
class CudaForceController {
public:
    CudaForceController();
    virtual ~CudaForceController();
    
    void setNumNodes(unsigned x);
    void setGravity(float x, float y, float z);
    void setMassBuf(CUDABuffer * x);
    void setImpulseBuf(CUDABuffer * x);
    void setMovenentRelativeToAir(const Vector3F & v);
    void resetMovenentRelativeToAir();
    
    void updateWind();
    void updateGravity(float dt);
    
    static Vector3F MovementRelativeToAir;
    static Vector3F WindDirection;
    static float WindMagnitude;
    
protected:

private:
    CUDABuffer * m_mass;
    CUDABuffer * m_impulse;
    float m_gravity[3];
    unsigned m_numNodes;
};
#endif        //  #ifndef CUDAFORCECONTROLLER_H

