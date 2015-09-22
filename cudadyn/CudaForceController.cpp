#include "CudaForceController.h"
#include <CUDABuffer.h>
#include <CudaBase.h>
#include <masssystem_impl.h>
#include <controlforce_impl.h>

Vector3F CudaForceController::MovementRelativeToAir = Vector3F::Zero;
Vector3F CudaForceController::WindDirection = Vector3F::XAxis;
float CudaForceController::WindMagnitude = 0.f;

CudaForceController::CudaForceController() 
{
    m_numNodes = 0;
    m_numActiveNodes = 0;
    m_gravity[0] = 0.f;
    m_gravity[1] = -9.81f;
    m_gravity[2] = 0.f;
    m_windSeed = 0;
    m_windTurbulence = 0.f;
}

CudaForceController::~CudaForceController() {}

void CudaForceController::setNumNodes(unsigned x)
{ m_numNodes = x; }

void CudaForceController::setNumActiveNodes(unsigned x)
{ m_numActiveNodes = x; }

void CudaForceController::setGravity(float x, float y, float z)
{ 
    m_gravity[0] = x;
    m_gravity[1] = y;
    m_gravity[2] = z;
    gravityforce::setGravity(m_gravity);
}

void CudaForceController::setMassBuf(CUDABuffer * x)
{ m_mass = x; }

void CudaForceController::setImpulseBuf(CUDABuffer * x)
{ m_impulse = x; }

void CudaForceController::setMovenentRelativeToAir(const Vector3F & v)
{ 
    MovementRelativeToAir = v; 
    MovementRelativeToAir.clamp(60.f);
}

void CudaForceController::resetMovenentRelativeToAir()
{ MovementRelativeToAir.setZero(); }

void CudaForceController::updateWind()
{
    if(m_numNodes < 1) return;
    void * mass = m_mass->bufferOnDevice();
    void * impulse = m_impulse->bufferOnDevice();
    masssystem::setVelocity((float3 *)impulse, 
                           (float *)mass,
                           -MovementRelativeToAir.x, -MovementRelativeToAir.y, -MovementRelativeToAir.z,
                           m_numActiveNodes);
                    
    if(WindMagnitude < 1e-5f) return;
    
    Vector3F maj = WindDirection * WindMagnitude;
    maj /= -50.f;
    
    Vector3F vu = WindDirection.perpendicular();
    Vector3F vv = WindDirection.cross(vu);
    vv.normalize();
    
    windforce::setWindVecs((float *)&maj,
                            (float *)&vu,
                            (float *)&vv);
    
    windforce::setWind((float3 *)impulse, 
                           (float *)mass,
                           m_windTurbulence * WindMagnitude,
                           m_windSeed,
                           m_numActiveNodes);
    CudaBase::CheckCudaError("force controller update wind");
}

void CudaForceController::updateGravity(float dt)
{
    if(m_numNodes < 1) return;
    void * mass = m_mass->bufferOnDevice();
    void * impusle = m_impulse->bufferOnDevice();
    
    gravityforce::addGravity((float3 *)impusle, 
                           (float *)mass,
                           dt,
                           m_numActiveNodes);
    CudaBase::CheckCudaError("force controller update gravity");
}

void CudaForceController::setWindSeed(unsigned x)
{ m_windSeed = (x * 16807UL) & 2147483647 ; }

void CudaForceController::setWindTurbulence(float x)
{ m_windTurbulence = x; }
//:~
