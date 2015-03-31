#ifndef CUDADYNAMICWORLD_H
#define CUDADYNAMICWORLD_H

#define CUDA_DYNAMIC_WORLD_MAX_NUM_OBJECTS 32

class GeoDrawer;
class CudaBroadphase;
class CudaNarrowphase;
class SimpleContactSolver;
class CudaTetrahedronSystem;
class DrawBvh;
class DrawNp;

class CudaDynamicWorld
{
public:
    CudaDynamicWorld();
    virtual ~CudaDynamicWorld();
    
    virtual void initOnDevice();
    
    void addTetrahedronSystem(CudaTetrahedronSystem * tetra);
    
    void stepPhysics(float dt);
    
    const unsigned numObjects() const;
    
    void setDrawer(GeoDrawer * drawer);
    
    CudaTetrahedronSystem * tetradedron(unsigned ind);
	
	void dbgDraw();
protected:

private:
    CudaBroadphase * m_broadphase;
    CudaNarrowphase * m_narrowphase;
    SimpleContactSolver * m_contactSolver;
    CudaTetrahedronSystem * m_objects[CUDA_DYNAMIC_WORLD_MAX_NUM_OBJECTS];
    unsigned m_numObjects;
    DrawBvh * m_dbgBvh;
	DrawNp * m_dbgNp;
};
#endif        //  #ifndef CUDADYNAMICWORLD_H

