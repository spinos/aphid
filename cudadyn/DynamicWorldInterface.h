#ifndef DYNAMICWORLDINTERFACE_H
#define DYNAMICWORLDINTERFACE_H
#include <AllMath.h>
class BaseBuffer;
class GeoDrawer;
class CudaDynamicWorld;
class TetrahedronSystem;
class CudaLinearBvh;
class DynamicWorldInterface {
public:
    DynamicWorldInterface();
    virtual ~DynamicWorldInterface();
    
    virtual void create(CudaDynamicWorld * world);
    
    void draw(CudaDynamicWorld * world);
    void draw(CudaDynamicWorld * world, GeoDrawer * drawer);
protected:

private:
    void draw(TetrahedronSystem * tetra);
    void showOverlappingPairs(CudaDynamicWorld * world, GeoDrawer * drawer);
    void showBvhHash(CudaDynamicWorld * world, GeoDrawer * drawer);
    void showBvhHash(CudaLinearBvh * bvh, GeoDrawer * drawer);
    void showContacts(CudaDynamicWorld * world, GeoDrawer * drawer);
    Vector3F tetrahedronCenter(Vector3F * p, unsigned * v, 
        unsigned * pntOffset, unsigned * indOffset, 
        unsigned i);
private:
    BaseBuffer * m_boxes;
    BaseBuffer * m_bvhHash;
    BaseBuffer * m_pairCache;
    BaseBuffer * m_tetPnt;
	BaseBuffer * m_tetInd;
	BaseBuffer * m_pointStarts;
	BaseBuffer * m_indexStarts;
	BaseBuffer * m_constraint;
	BaseBuffer * m_contactPairs;
	BaseBuffer * m_contact;
	BaseBuffer * m_pairsHash;
	BaseBuffer * m_linearVelocity;
	BaseBuffer * m_angularVelocity;
};
#endif        //  #ifndef DYNAMICWORLDINTERFACE_H

