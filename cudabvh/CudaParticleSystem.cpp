#include "CudaParticleSystem.h"

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
    m_numParticles = n;
}
    
void CudaParticleSystem::initOnDevice()
{
    m_X = new CUDABuffer;
    m_X->create(numParticles() * 12);
    m_V = new CUDABuffer;
    m_V->create(numParticles() * 12);
    m_F = new CUDABuffer;
    m_F->create(numParticles() * 12);
    
    m_X->hostToDevice(m_hostX->data(), m_X->bufferSize());
    m_V->hostToDevice(m_hostV->data(), m_V->bufferSize());
    m_F->hostToDevice(m_hostF->data(), m_F->bufferSize());
}

const unsigned CudaParticleSystem::numParticles() const
{ return m_numParticles; }

void * CudaParticleSystem::position()
{ return m_hostX->data(); }

void * CudaParticleSystem::positionOnDevice()
{ return m_X->bufferOnDevice(); }

void * CudaParticleSystem::velocity()
{ return m_hostV->data(); }

void * CudaParticleSystem::velocityOnDevice()
{ return m_V->bufferOnDevice(); }
	
void * CudaParticleSystem::force()
{ return m_hostF->data(); }

void * CudaParticleSystem::forceOnDevice()
{ return m_F->bufferOnDevice(); }

void CudaParticleSystem::deviceToHost()
{ m_X->deviceToHost(m_hostX->data(), m_X->bufferSize()); }

void CudaParticleSystem::update() 
{
    computeForce();
    integrate();
}

void CudaParticleSystem::computeForce() {}
void CudaParticleSystem::integrate() {}

