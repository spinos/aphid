#ifndef CUDADYNAMICWORLD_H
#define CUDADYNAMICWORLD_H
#include <vector>
#include "DynGLobal.h"
#include <Vector3F.h>
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
class H5FieldOut;
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
    void updateSystem(float dt);
    void integrate(float dt);
    void sendXToHost();
    void readVelocityCache();
	void reset();
	void updateWind();
	void updateSpeedLimit(float x);
    
    bool isToSaveCache() const;
    void setToSaveCache(bool x);
    virtual void saveCache();
    void beginCache();
    bool allFramesCached() const;
    
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
    static Vector3F MovementRelativeToAir;
protected:
    void resetMovenentRelativeToAir();
	void updateMovenentRelativeToAir();
private:
    std::string objName(int i) const;
private:
    std::vector<unsigned > m_activeObjectInds;
    CudaBroadphase * m_broadphase;
    CudaNarrowphase * m_narrowphase;
    SimpleContactSolver * m_contactSolver;
    CudaMassSystem * m_objects[CUDA_DYNAMIC_WORLD_MAX_NUM_OBJECTS];
    H5FieldOut * m_positionFile;
    unsigned m_numObjects;
    bool m_enableSaveCache;
    bool m_finishedCaching;
};
#endif        //  #ifndef CUDADYNAMICWORLD_H