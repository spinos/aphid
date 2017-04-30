#ifndef CUDADYNAMICSYSTEM_H
#define CUDADYNAMICSYSTEM_H

#include "CollisionQuery.h"

class CUDABuffer;

class CudaDynamicSystem : public CollisionQuery {
public:
    CudaDynamicSystem();
    virtual ~CudaDynamicSystem();
    
    virtual void initOnDevice();
    virtual void update(float dt);
    
protected:
    void setNumPoints(unsigned n);
    const unsigned numPoints() const;
    
    void * positionOnDevice();	
	void * velocityOnDevice();
	void * forceOnDevice();
	
	CUDABuffer * X();
	CUDABuffer * V();
	CUDABuffer * F();
	
private:
    void integrate(float dt);
    
private:
    CUDABuffer * m_X;
	CUDABuffer * m_V;
    CUDABuffer * m_F;
    unsigned m_numPoints;
};
#endif        //  #ifndef CUDADYNAMICSYSTEM_H

