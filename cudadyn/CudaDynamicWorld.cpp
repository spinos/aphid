#include "CudaDynamicWorld.h"
#include <CudaBroadphase.h>
#include <CudaNarrowphase.h>
#include <SimpleContactSolver.h>
#include <CudaTetrahedronSystem.h>
#include <CudaBase.h>
#include <BvhBuilder.h>
#include <WorldDbgDraw.h>

WorldDbgDraw * CudaDynamicWorld::DbgDrawer = 0;

CudaDynamicWorld::CudaDynamicWorld() 
{
    m_broadphase = new CudaBroadphase;
    m_narrowphase = new CudaNarrowphase;
	m_contactSolver = new SimpleContactSolver;
	m_numObjects = 0;
}

CudaDynamicWorld::~CudaDynamicWorld()
{
    delete m_broadphase;
    delete m_narrowphase;
    delete m_contactSolver;
}

const unsigned CudaDynamicWorld::numObjects() const
{ return m_numObjects; }

CudaTetrahedronSystem * CudaDynamicWorld::tetradedron(unsigned ind) const
{ return m_objects[ind]; }

CudaBroadphase * CudaDynamicWorld::broadphase() const
{ return m_broadphase; }

CudaNarrowphase * CudaDynamicWorld::narrowphase() const
{ return m_narrowphase; }

SimpleContactSolver * CudaDynamicWorld::contactSolver() const
{ return m_contactSolver; }

void CudaDynamicWorld::setBvhBuilder(BvhBuilder * builder)
{ CudaLinearBvh::Builder = builder; }

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
	
	CudaLinearBvh::Builder->initOnDevice();
    
    unsigned i;
	for(i=0; i < m_numObjects; i++) m_objects[i]->initOnDevice();
	
    m_broadphase->initOnDevice();
    m_narrowphase->initOnDevice();
	m_contactSolver->initOnDevice();
}

void CudaDynamicWorld::stepPhysics(float dt)
{
    collide();
	integrate(dt);
}

void CudaDynamicWorld::collide()
{
    if(m_numObjects < 1) return;
	unsigned i;
	for(i=0; i < m_numObjects; i++) m_objects[i]->update();
    
    //std::cout<<"aft bvh ";
   // CudaBase::CheckCudaError("bvh update");
        
	m_broadphase->computeOverlappingPairs();
    
    //std::cout<<"aft bph ";
    //CudaBase::CheckCudaError("bph update");
	
	m_narrowphase->computeContacts(m_broadphase->overlappingPairBuf(), 
	                                m_broadphase->numOverlappingPairs());
	
   // std::cout<<"aft contact ";
    //CudaBase::CheckCudaError("contact");
	
	m_contactSolver->solveContacts(m_narrowphase->numContacts(),
									m_narrowphase->contactBuffer(),
									m_narrowphase->contactPairsBuffer(),
									m_narrowphase->objectBuffer());
    
    //std::cout<<"aft contact solve ";
    //CudaBase::CheckCudaError("contact solve");
}

void CudaDynamicWorld::integrate(float dt)
{
    for(unsigned i=0; i < m_numObjects; i++) m_objects[i]->integrate(dt);
}

const unsigned CudaDynamicWorld::numContacts() const
{ return m_narrowphase->numContacts(); }

void CudaDynamicWorld::reset()
{
    if(m_numObjects < 1) return;
    m_narrowphase->resetToInitial();
}

void CudaDynamicWorld::sendXToHost()
{
    const unsigned nobj = numObjects();
    if(nobj<1) return;
    
    // cudaDeviceSynchronize();

	cudaEvent_t start_event, stop_event;
        
	cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);

	cudaEventRecord(start_event, 0);
    
	 unsigned i;
     for(i=0; i< nobj; i++) {
        tetradedron(i)->sendXToHost();
        m_objects[i]->sendDbgToHost();
     }
	
	m_broadphase->sendDbgToHost();
		
	cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
	
	float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
#if PRINT_TRANSACTION_TIME	
	std::cout<<" device-host transaction time: "<<elapsed_time<<" milliseconds\n";
#endif
	cudaEventDestroy( start_event ); 
	cudaEventDestroy( stop_event );
}

void CudaDynamicWorld::dbgDraw()
{
    if(!CudaDynamicWorld::DbgDrawer) return;
    const unsigned nobj = numObjects();
    if(nobj<1) return;

#if DRAW_BVH_HASH
    for(unsigned i=0; i< nobj; i++) {
        DbgDrawer->showBvhHash(m_objects[i]);
	}
#endif
}

