#include "CudaForceController.h"
#include <CUDABuffer.h>
#include <CudaBase.h>
#include <masssystem_impl.h>

CudaForceController::CudaForceController() 
{
    m_numNodes = 0;
    m_gravity[0] = 0.f;
    m_gravity[1] = -9.81f;
    m_gravity[2] = 0.f;
}

CudaForceController::~CudaForceController() {}

void CudaForceController::setNumNodes(unsigned x)
{ m_numNodes = x; }

void CudaForceController::setGravity(float x, float y, float z)
{ 
    m_gravity[0] = x;
    m_gravity[1] = y;
    m_gravity[2] = z;
    masssystem::setGravity(m_gravity);
}

void CudaForceController::setMassBuf(CUDABuffer * x)
{ m_mass = x; }

void CudaForceController::setImpulseBuf(CUDABuffer * x)
{ m_impulse = x; }

void CudaForceController::updateGravity(float dt)
{
    if(m_numNodes < 1) return;
    void * mass = m_mass->bufferOnDevice();
    void * impusle = m_impulse->bufferOnDevice();
    
    masssystem::addGravity((float3 *)impusle, 
                           (float *)mass,
                           dt,
                           m_numNodes);
    CudaBase::CheckCudaError("force controller update gravity");
}
//:~
