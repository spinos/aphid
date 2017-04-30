#ifndef CUDAPARTICLESYSTEM_H
#define CUDAPARTICLESYSTEM_H

#include "bvh_common.h"
#include <CudaDynamicSystem.h>
class CUDABuffer;
class BaseBuffer;

class CudaParticleSystem : public CudaDynamicSystem
{
public:
	CudaParticleSystem();
	virtual ~CudaParticleSystem();
	
	void createParticles(uint n);
	virtual void initOnDevice();
	
	const unsigned numParticles() const;
	
	virtual void update(float dt);
	void deviceToHost();
	
	void * position();
	void * velocity();
	void * force();

protected:
    
private:
	void computeForce();
    void integrate(float dt);
	
private:
    BaseBuffer * m_hostX;
    BaseBuffer * m_hostV;
    BaseBuffer * m_hostF;
};


#endif        //  #ifndef CUDAPARTICLESYSTEM_H

