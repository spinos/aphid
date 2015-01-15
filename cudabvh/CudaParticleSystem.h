#ifndef CUDAPARTICLESYSTEM_H
#define CUDAPARTICLESYSTEM_H

#include "bvh_common.h"

class CUDABuffer;
class BaseBuffer;

class CudaParticleSystem
{
public:
	CudaParticleSystem();
	virtual ~CudaParticleSystem();
	
	void createParticles(uint n);
	void initOnDevice();
	
	const unsigned numParticles() const;
	
	void update(float dt);
	void deviceToHost();
	
	void * position();
	void * positionOnDevice();
	
	void * velocity();
	void * velocityOnDevice();
	
	void * force();
	void * forceOnDevice();
	
protected:
    
private:
	void computeForce();
    void integrate(float dt);
	
private:
	CUDABuffer * m_X;
	CUDABuffer * m_V;
    CUDABuffer * m_F;
    
    BaseBuffer * m_hostX;
    BaseBuffer * m_hostV;
    BaseBuffer * m_hostF;
	
	unsigned m_numParticles;
};


#endif        //  #ifndef CUDAPARTICLESYSTEM_H

