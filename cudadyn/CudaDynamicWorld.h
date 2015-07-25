#ifndef CUDADYNAMICWORLD_H
#define CUDADYNAMICWORLD_H

#include "DynGLobal.h"
class CudaBroadphase;
class CudaNarrowphase;
class SimpleContactSolver;
class TriangleSystem;
class BvhTetrahedronSystem;
class BvhBuilder;
class WorldDbgDraw;
class BvhTriangleSystem;
class CudaMassSystem;
class CudaLinearBvh;
class IVelocityFile;
class CudaDynamicWorld
{
public:
    CudaDynamicWorld();
    virtual ~CudaDynamicWorld();
    
    virtual void initOnDevice();
    
	void setBvhBuilder(BvhBuilder * builder);
    void addTetrahedronSystem(BvhTetrahedronSystem * tetra);
    void addTriangleSystem(BvhTriangleSystem * tri);
    
    void stepPhysics(float dt);
    
    void collide();
    void integrate(float dt);
    void sendXToHost();
    void readVelocityCache();
	void reset();
    
    const unsigned numObjects() const;
    CudaLinearBvh * bvhObject(unsigned idx) const;
	CudaMassSystem * object(unsigned idx) const;
    
    CudaBroadphase * broadphase() const;
	CudaNarrowphase * narrowphase() const;
	SimpleContactSolver * contactSolver() const;
	const unsigned numContacts() const;
    const unsigned totalNumPoints() const;
    void dbgDraw();
    
    static WorldDbgDraw * DbgDrawer;
    static IVelocityFile * VelocityCache;
protected:
	
private:
    CudaBroadphase * m_broadphase;
    CudaNarrowphase * m_narrowphase;
    SimpleContactSolver * m_contactSolver;
    CudaMassSystem * m_objects[CUDA_DYNAMIC_WORLD_MAX_NUM_OBJECTS];
    unsigned m_numObjects;
};
#endif        //  #ifndef CUDADYNAMICWORLD_H

