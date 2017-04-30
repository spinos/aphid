#ifndef DYNAMICWORLDINTERFACE_H
#define DYNAMICWORLDINTERFACE_H
#include <AllMath.h>
#include <DynGlobal.h>
class BaseBuffer;
class GeoDrawer;
class CudaDynamicWorld;
class CudaNarrowphase;
class TetrahedronSystem;
class CudaLinearBvh;
class SimpleContactSolver;
class TriangleSystem;
class MassSystem;
class DynamicWorldInterface {
public:
    DynamicWorldInterface();
    virtual ~DynamicWorldInterface();
    
    virtual void create(CudaDynamicWorld * world);
    void updateWindSpeed(float x);
    void updateWindVec(const Vector3F & v);
	void changeMaxDisplayLevel(int d);
    void draw(CudaDynamicWorld * world, GeoDrawer * drawer);
    // void draw(CudaDynamicWorld * world, GeoDrawer * drawer);
    void drawFaulty(CudaDynamicWorld * world, GeoDrawer * drawer);
    bool verifyData(CudaDynamicWorld * world);
protected:

private:
    void drawTetrahedron(TetrahedronSystem * tetra, GeoDrawer * drawer, int ind);
    void drawSystem(MassSystem * tri);
#if DRAW_BPH_PAIRS
    void showOverlappingPairs(CudaDynamicWorld * world, GeoDrawer * drawer);
#endif
#if DRAW_BVH_HASH
    void showBvhHash(CudaDynamicWorld * world, GeoDrawer * drawer);
    void showBvhHash(CudaLinearBvh * bvh, GeoDrawer * drawer);
#endif

#if DRAW_NPH_CONTACT
    void showContacts(CudaDynamicWorld * world, GeoDrawer * drawer);
#endif

#if DRAW_BVH_HIERARCHY
	void showBvhHierarchy(CudaDynamicWorld * world, GeoDrawer * drawer);
	void showBvhHierarchy(CudaLinearBvh * bvh, GeoDrawer * drawer);
#endif

    bool checkContact(unsigned n);
    bool checkDegenerated(unsigned n);
    void printContact(unsigned n);
    bool checkConstraint(SimpleContactSolver * solver, unsigned n);
    void printConstraint(SimpleContactSolver * solver, unsigned n);
    bool checkConvergent(SimpleContactSolver * solver, unsigned n);
    void printContactPairHash(SimpleContactSolver * solver, unsigned numContacts);
    void printFaultPair(CudaDynamicWorld * world);
    void storeModels(CudaNarrowphase * narrowphase);
    void showFaultyPair(CudaDynamicWorld * world, GeoDrawer * drawer);
private:
	int m_maxDisplayLevel;
    unsigned m_faultyPair[2];
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
	BaseBuffer * m_deltaJ;
};
#endif        //  #ifndef DYNAMICWORLDINTERFACE_H

