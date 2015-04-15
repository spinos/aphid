#ifndef DYNAMICWORLDINTERFACE_H
#define DYNAMICWORLDINTERFACE_H
#include <AllMath.h>
class BaseBuffer;
class GeoDrawer;
class CudaDynamicWorld;
class CudaNarrowphase;
class TetrahedronSystem;
class CudaLinearBvh;
class SimpleContactSolver;
class DynamicWorldInterface {
public:
    DynamicWorldInterface();
    virtual ~DynamicWorldInterface();
    
    virtual void create(CudaDynamicWorld * world);
    
    void draw(CudaDynamicWorld * world);
    void draw(CudaDynamicWorld * world, GeoDrawer * drawer);
    void drawFaulty(CudaDynamicWorld * world, GeoDrawer * drawer);
    bool verifyData(CudaDynamicWorld * world);
protected:

private:
    void draw(TetrahedronSystem * tetra);
    void showOverlappingPairs(CudaDynamicWorld * world, GeoDrawer * drawer);
    void showBvhHash(CudaDynamicWorld * world, GeoDrawer * drawer);
    void showBvhHash(CudaLinearBvh * bvh, GeoDrawer * drawer);
    void showContacts(CudaDynamicWorld * world, GeoDrawer * drawer);
    bool checkContact(unsigned n);
    bool checkDegenerated(unsigned n);
    void printContact(unsigned n);
    bool checkConstraint(SimpleContactSolver * solver, unsigned n);
    void printContactPairHash(SimpleContactSolver * solver, unsigned numContacts);
    void printFaultPair(CudaDynamicWorld * world);
    void storeModels(CudaNarrowphase * narrowphase);
    void showFaultyPair(CudaDynamicWorld * world, GeoDrawer * drawer);
private:
    unsigned m_faultyPair[2];
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
	BaseBuffer * m_mass;
	BaseBuffer * m_split;
};
#endif        //  #ifndef DYNAMICWORLDINTERFACE_H

