#ifndef CUDAFORCECONTROLLER_H
#define CUDAFORCECONTROLLER_H
#include <Vector3F.h>
class CUDABuffer;
class CudaForceController {
public:
    CudaForceController();
    virtual ~CudaForceController();
    
    void setNumNodes(unsigned x);
    void setNumActiveNodes(unsigned x);
    void setGravity(float x, float y, float z);
    void setMassBuf(CUDABuffer * x);
    void setImpulseBuf(CUDABuffer * x);
    void setMovenentRelativeToAir(const Vector3F & v);
    void resetMovenentRelativeToAir();
    void setWindSeed(unsigned x);
    void setWindTurbulence(float x);
    
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
    unsigned m_numNodes, m_numActiveNodes;
    unsigned m_windSeed;
    float m_windTurbulence;
};
#endif        //  #ifndef CUDAFORCECONTROLLER_H

