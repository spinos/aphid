#include "CudaDynamicWorld.h"
#include <CudaBroadphase.h>
#include <CudaNarrowphase.h>
#include <SimpleContactSolver.h>
#include <CudaTetrahedronSystem.h>
#include <CudaBase.h>
#include <DrawBvh.h>
#include <DrawNp.h>

CudaDynamicWorld::CudaDynamicWorld() 
{
    m_broadphase = new CudaBroadphase;
    m_narrowphase = new CudaNarrowphase;
	m_contactSolver = new SimpleContactSolver;
	m_numObjects = 0;
	
	m_dbgBvh = new DrawBvh;
	m_dbgNp = new DrawNp;
}

CudaDynamicWorld::~CudaDynamicWorld()
{
    delete m_broadphase;
    delete m_narrowphase;
    delete m_contactSolver;
}

const unsigned CudaDynamicWorld::numObjects() const
{ return m_numObjects; }

void CudaDynamicWorld::setDrawer(GeoDrawer * drawer)
{ 
	m_dbgBvh->setDrawer(drawer); 
	m_dbgNp->setDrawer(drawer);
}

CudaTetrahedronSystem * CudaDynamicWorld::tetradedron(unsigned ind)
{ return m_objects[ind]; }

void CudaDynamicWorld::addTetrahedronSystem(CudaTetrahedronSystem * tetra)
{
    if(m_numObjects == CUDA_DYNAMIC_WORLD_MAX_NUM_OBJECTS) return;
    
    m_objects[m_numObjects] = tetra;
    m_numObjects++;
    
    m_broadphase->addBvh(tetra);
    m_narrowphase->addTetrahedronSystem(tetra);
}

void CudaDynamicWorld::initOnDevice()
{
    if(m_numObjects < 1) return;
    CudaBase::SetDevice();
    
    unsigned i;
	for(i=0; i < m_numObjects; i++) m_objects[i]->initOnDevice();
	
    m_broadphase->initOnDevice();
    m_narrowphase->initOnDevice();
	m_contactSolver->initOnDevice();
}

void CudaDynamicWorld::stepPhysics(float dt)
{
    if(m_numObjects < 1) return;
	unsigned i;
	for(i=0; i < m_numObjects; i++) m_objects[i]->update();
	m_broadphase->computeOverlappingPairs();
	
	m_narrowphase->computeContacts(m_broadphase->overlappingPairBuf(), 
	                                m_broadphase->numUniquePairs());
	
	m_contactSolver->solveContacts(m_narrowphase->numContacts(),
									m_narrowphase->contactBuffer(),
									m_narrowphase->contactPairsBuffer(),
									m_narrowphase->objectBuffer());
									
	for(i=0; i < m_numObjects; i++) m_objects[i]->integrate(dt);
}

void CudaDynamicWorld::dbgDraw()
{
	std::cout<<" num overlapping pairs "<<m_broadphase->numUniquePairs();
	// m_dbgBvh->showOverlappingPairs(m_broadphase);
	
	unsigned i;
	for(i=0; i < m_numObjects; i++) {
	    // m_dbgBvh->showHash(m_objects[i]);
	}
	std::cout<<" num contact pairs "<<m_narrowphase->numContacts();
	
	// m_dbgNp->showConstraint(m_contactSolver, m_narrowphase);
	std::cout<<" mem "<<CudaBase::MemoryUsed<<"\n";
}
