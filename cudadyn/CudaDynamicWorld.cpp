#include "CudaDynamicWorld.h"
#include <CudaBroadphase.h>
#include <CudaNarrowphase.h>
#include <SimpleContactSolver.h>
#include <BvhTetrahedronSystem.h>
#include <BvhTriangleSystem.h>
#include <CudaBase.h>
#include <BvhBuilder.h>
#include <WorldDbgDraw.h>
#include <IVelocityFile.h>
#include <CudaDbgLog.h>

CudaDbgLog wdlg("world.txt");
WorldDbgDraw * CudaDynamicWorld::DbgDrawer = 0;
IVelocityFile * CudaDynamicWorld::VelocityCache = 0;

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

CudaBroadphase * CudaDynamicWorld::broadphase() const
{ return m_broadphase; }

CudaNarrowphase * CudaDynamicWorld::narrowphase() const
{ return m_narrowphase; }

SimpleContactSolver * CudaDynamicWorld::contactSolver() const
{ return m_contactSolver; }

void CudaDynamicWorld::setBvhBuilder(BvhBuilder * builder)
{ CudaLinearBvh::Builder = builder; }

void CudaDynamicWorld::addTetrahedronSystem(BvhTetrahedronSystem * tetra)
{
    if(m_numObjects == CUDA_DYNAMIC_WORLD_MAX_NUM_OBJECTS) return;
    
    m_objects[m_numObjects] = tetra;
    m_numObjects++;
    
    m_broadphase->addBvh(tetra);
    m_narrowphase->addMassSystem(tetra);
}

void CudaDynamicWorld::addTriangleSystem(BvhTriangleSystem * tri)
{
    if(m_numObjects == CUDA_DYNAMIC_WORLD_MAX_NUM_OBJECTS) return;
    
    m_objects[m_numObjects] = tri;
    m_numObjects++;
    
    m_broadphase->addBvh(tri);
    m_narrowphase->addMassSystem(tri);
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
	for(i=0; i < m_numObjects; i++) bvhObject(i)->update();
    CudaBase::CheckCudaError("bvh update");
        
	m_broadphase->computeOverlappingPairs();
    // CudaBase::CheckCudaError("broadphase update");
	
	m_narrowphase->computeContacts(m_broadphase->overlappingPairBuf(), 
	                                m_broadphase->numOverlappingPairs());
    // CudaBase::CheckCudaError("narrowphase update");
	
	m_contactSolver->solveContacts(m_narrowphase->numContacts(),
									m_narrowphase->contactBuffer(),
									m_narrowphase->contactPairsBuffer(),
									m_narrowphase->objectBuffer());
    // CudaBase::CheckCudaError("solve contact");
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
    if(!VelocityCache) return;
   
    VelocityCache->frameBegin();
    VelocityCache->readFrameVelocity();
    VelocityCache->nextFrame();
    
    m_narrowphase->setAnchoredVelocity(VelocityCache->velocities());
}

void CudaDynamicWorld::sendXToHost()
{
    const unsigned nobj = numObjects();
    if(nobj<1) return;
    
    // cudaDeviceSynchronize();
#if 0
	cudaEvent_t start_event, stop_event;
    
	cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);

	cudaEventRecord(start_event, 0);
#endif   
	 unsigned i;
     for(i=0; i< nobj; i++) {
		m_objects[i]->sendXToHost();
        CudaLinearBvh * bvh = bvhObject(i);
        bvh->sendDbgToHost();
     }
	
	m_broadphase->sendDbgToHost();
#if 0		
	cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
	
	float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
#if PRINT_TRANSACTION_TIME	
	std::cout<<" device-host transaction time: "<<elapsed_time<<" milliseconds\n";
#endif
	cudaEventDestroy( start_event ); 
	cudaEventDestroy( stop_event );
#endif
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

#if DRAW_BVH_HIERARCHY
    for(unsigned i=0; i< nobj; i++) {
        DbgDrawer->showBvhHierarchy(bvhObject(i));
	}
#endif
}

CudaLinearBvh * CudaDynamicWorld::bvhObject(unsigned idx) const
{ return dynamic_cast<CudaLinearBvh *>(m_objects[idx]); }

CudaMassSystem * CudaDynamicWorld::object(unsigned idx) const
{ return m_objects[idx]; }

const unsigned CudaDynamicWorld::totalNumPoints() const
{ return m_narrowphase->numPoints(); }

void CudaDynamicWorld::readVelocityCache()
{
    if(!VelocityCache) return;

    if(VelocityCache->isFrameBegin()) m_narrowphase->resetToInitial();
    VelocityCache->readFrameVelocity();
    if(VelocityCache->isFrameEnd()) VelocityCache->frameBegin();
    else VelocityCache->nextFrame();
    
   //wdlg.writeVec3(VelocityCache->velocityBuf(), 
	//				VelocityCache->numPoints(), 
	//				" Va ", CudaDbgLog::FAlways);
    
    m_narrowphase->setAnchoredVelocity(VelocityCache->velocities());
}
//:~
