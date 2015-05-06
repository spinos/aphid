#ifndef CUDADYNAMICWORLD_H
#define CUDADYNAMICWORLD_H

#include "DynGLobal.h"

class CudaBroadphase;
class CudaNarrowphase;
class SimpleContactSolver;
class CudaTetrahedronSystem;
class BvhBuilder;
class WorldDbgDraw;
class CudaDynamicWorld
{
public:
    CudaDynamicWorld();
    virtual ~CudaDynamicWorld();
    
    virtual void initOnDevice();
    
	void setBvhBuilder(BvhBuilder * builder);
    void addTetrahedronSystem(CudaTetrahedronSystem * tetra);
    
    void stepPhysics(float dt);
    
    void collide();
    void integrate(float dt);
    void sendXToHost();
	void reset();
    
    const unsigned numObjects() const;
    
    CudaTetrahedronSystem * tetradedron(unsigned ind) const;
	CudaBroadphase * broadphase() const;
	CudaNarrowphase * narrowphase() const;
	SimpleContactSolver * contactSolver() const;
	const unsigned numContacts() const;
    
    void dbgDraw();
    
    static WorldDbgDraw * DbgDrawer;
protected:
	
private:
    CudaBroadphase * m_broadphase;
    CudaNarrowphase * m_narrowphase;
    SimpleContactSolver * m_contactSolver;
    CudaTetrahedronSystem * m_objects[CUDA_DYNAMIC_WORLD_MAX_NUM_OBJECTS];
    unsigned m_numObjects;
};
#endif        //  #ifndef CUDADYNAMICWORLD_H

