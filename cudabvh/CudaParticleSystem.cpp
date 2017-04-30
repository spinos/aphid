#include "CudaParticleSystem.h"
#include "particleSystem_implement.h"
#include <CUDABuffer.h>
#include <BaseBuffer.h>

CudaParticleSystem::CudaParticleSystem() {}
CudaParticleSystem::~CudaParticleSystem() {}

void CudaParticleSystem::createParticles(uint n)
{
    m_hostX = new BaseBuffer;
    m_hostX->create(n * 12);
    m_hostV = new BaseBuffer;
    m_hostV->create(n * 12);
    m_hostF = new BaseBuffer;
    m_hostF->create(n * 12);
    setNumPoints(n);
    setNumPrimitives(n);
}
    
void CudaParticleSystem::initOnDevice()
{
    CudaDynamicSystem::initOnDevice();
    
    X()->hostToDevice(m_hostX->data(), X()->bufferSize());
    V()->hostToDevice(m_hostV->data(), V()->bufferSize());
    F()->hostToDevice(m_hostF->data(), F()->bufferSize());
}

const unsigned CudaParticleSystem::numParticles() const
{ return numPoints(); }

void * CudaParticleSystem::position()
{ return m_hostX->data(); }

void * CudaParticleSystem::velocity()
{ return m_hostV->data(); }
	
void * CudaParticleSystem::force()
{ return m_hostF->data(); }

void CudaParticleSystem::deviceToHost()
{ X()->deviceToHost(m_hostX->data()); }

void CudaParticleSystem::update(float dt) 
{
    computeForce();
    CudaDynamicSystem::update(dt);
}

void CudaParticleSystem::computeForce() 
{ particleSystemSimpleGravityForce((float3 *)forceOnDevice(), numParticles()); }

