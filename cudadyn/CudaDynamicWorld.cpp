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
#include <cuFemTetrahedron_implement.h>
#include <H5FieldOut.h>
#include <AField.h>
#include "CudaForceController.h"

WorldDbgDraw * CudaDynamicWorld::DbgDrawer = 0;
IVelocityFile * CudaDynamicWorld::VelocityCache = 0;

CudaDynamicWorld::CudaDynamicWorld() 
{
    m_broadphase = new CudaBroadphase;
    m_narrowphase = new CudaNarrowphase;
	m_contactSolver = new SimpleContactSolver;
	m_numObjects = 0;
	m_numSimulationSteps = 0;
	m_totalEnergy = 0.f;
    m_enableSaveCache = false;
    m_finishedCaching = false;
    m_positionFile = new H5FieldOut;
    if(!m_positionFile->create("./position.tmp"))
        std::cout<<"\n error: dynamic world cannot create position cache file!\n";
    m_controller = new CudaForceController;
    m_isPendingReset = false;
}

CudaDynamicWorld::~CudaDynamicWorld()
{
    delete m_broadphase;
    delete m_narrowphase;
    delete m_contactSolver;
    delete m_controller;
}

const unsigned CudaDynamicWorld::numObjects() const
{ return m_numObjects; }

CudaBroadphase * CudaDynamicWorld::broadphase() const
{ return m_broadphase; }

CudaNarrowphase * CudaDynamicWorld::narrowphase() const
{ return m_narrowphase; }

SimpleContactSolver * CudaDynamicWorld::contactSolver() const
{ return m_contactSolver; }

CudaForceController * CudaDynamicWorld::controller() const
{ return m_controller; }

void CudaDynamicWorld::setBvhBuilder(BvhBuilder * builder)
{ CudaLinearBvh::Builder = builder; }

void CudaDynamicWorld::addTetrahedronSystem(BvhTetrahedronSystem * tetra)
{
    if(m_numObjects == CUDA_DYNAMIC_WORLD_MAX_NUM_OBJECTS) return;
    
    m_objects[m_numObjects] = tetra;
    m_activeObjectInds.push_back(m_numObjects);
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
	for(i=0; i < m_numObjects; i++)
        m_objects[i]->initOnDevice();
	
    m_broadphase->initOnDevice();
    m_narrowphase->initOnDevice();
	m_contactSolver->initOnDevice();
	std::cout<<"\n cuda dynamice world initialized"
	<<"\n used "<<CudaBase::MemoryUsed<<" byte memory"
	<<"\n";
	
	m_contactSolver->setSpeedLimit(50.f);
	m_controller->setNumNodes( m_narrowphase->numPoints() );
	std::cout<<"\n n active nodes "<<m_narrowphase->numActiveNodes();
	m_controller->setNumActiveNodes( m_narrowphase->numActiveNodes() );
	m_controller->setMassBuf( m_narrowphase->objectBuffer()->m_mass );
	m_controller->setImpulseBuf( m_narrowphase->objectBuffer()->m_linearImpulse );
	m_controller->setGravity(0.f, -9.81f, 0.f);
}

void CudaDynamicWorld::stepPhysics(float dt)
{
    m_numSimulationSteps++;
    if( allSleeping() ) return;
// add impulse
    updateWind();
    updateGravity(dt);
// resolve contact, update impulse of collision
    collide();
// update system by impulse
    updateSystem(dt);
// update position
	integrate(dt);
}

void CudaDynamicWorld::updateWind()
{ 
    m_controller->setWindSeed(m_numSimulationSteps);
    m_controller->updateWind(); 
}

void CudaDynamicWorld::updateGravity(float dt)
{ m_controller->updateGravity(dt); }

void CudaDynamicWorld::collide()
{
    unsigned i;
    for(i=0; i < m_numObjects; i++) bvhObject(i)->updateBvhImpulseBased();
       
	m_broadphase->computeOverlappingPairs();

	m_narrowphase->computeContacts(m_broadphase->overlappingPairBuf(), 
	                                m_broadphase->numOverlappingPairs(),
                                    m_totalEnergy > 0.f);
                                    //false);

	m_contactSolver->solveContacts(m_narrowphase->numContacts(),
									m_narrowphase->contactBuffer(),
									m_narrowphase->contactPairsBuffer(),
									m_narrowphase->objectBuffer());
}

void CudaDynamicWorld::updateSystem(float dt)
{
    unsigned i;
    for(i=0; i < m_numObjects; i++) m_objects[i]->updateSystem(dt);
}

void CudaDynamicWorld::integrate(float dt)
{ m_narrowphase->upatePosition(dt); }

const unsigned CudaDynamicWorld::numContacts() const
{ return m_narrowphase->numContacts(); }

void CudaDynamicWorld::reset()
{
    if(!m_isPendingReset) return;
    setToSaveCache(false);
    if(m_numObjects < 1) return;
    if(VelocityCache) {
        delete VelocityCache;
        VelocityCache = 0;
    }
    m_controller->resetMovenentRelativeToAir();
    wakeUpAll();
    resetAll();
    m_totalEnergy = 0.f;
    m_narrowphase->reset();
    m_isPendingReset = false;
}

void CudaDynamicWorld::sendXToHost()
{
    const unsigned nobj = numObjects();
    if(nobj<1) return;
    
#if 0
	cudaEvent_t start_event, stop_event;
    
	cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);

	cudaEventRecord(start_event, 0);
#endif   
	 unsigned i;
     for(i=0; i< nobj; i++) {
		if(m_objects[i]->isSleeping()) continue;
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
    if(VelocityCache->isOutOfRange()) {
        m_controller->resetMovenentRelativeToAir();
        VelocityCache->nextFrame();
        return;
    }

    VelocityCache->readFrameTranslationalVelocity();
    VelocityCache->readFrameVelocity();
    VelocityCache->nextFrame();

	m_narrowphase->setAnchoredVelocity(VelocityCache->velocities());

    Vector3F air = *VelocityCache->translationalVelocity();
    air.reverse();
    m_controller->setMovenentRelativeToAir(air);
}

bool CudaDynamicWorld::isToSaveCache() const
{ return m_enableSaveCache; }

void CudaDynamicWorld::setToSaveCache(bool x)
{ 
    m_enableSaveCache = x; 
    m_finishedCaching = false;
}

void CudaDynamicWorld::saveCache()
{
    if(!m_enableSaveCache) return;
    if(!VelocityCache) return;
    //if(VelocityCache->isOutOfRange()) return;
    if(VelocityCache->currentFrame() > VelocityCache->LastFrame+1) {
        m_finishedCaching = true;
        return;
    }
    
    std::cout<<"\n caching frame "<<VelocityCache->currentFrame()-1;
    
    std::vector<unsigned >::const_iterator it = m_activeObjectInds.begin();
    for(;it!=m_activeObjectInds.end();++it) {
        AField * f = m_positionFile->fieldByName(objName(*it));
        TypedBuffer * buf = f->namedChannel("P");
        buf->copyFrom(m_objects[*it]->hostX(), m_objects[*it]->numPoints() * 12);
    }
    m_positionFile->writeFrame(VelocityCache->currentFrame()-1);
}

void CudaDynamicWorld::beginCache()
{
    m_finishedCaching = false;
    if(!m_enableSaveCache) return;
    if(!VelocityCache) return;
    std::cout<<"\n dynamic world begin cache frames ("<<VelocityCache->FirstFrame
    <<","<<VelocityCache->LastFrame<<")";
    m_positionFile->writeFrameRange(VelocityCache);
	
// todo m_positionFile->addFlt3Attribute(".translate");
	
    std::vector<unsigned >::const_iterator it = m_activeObjectInds.begin();
    for(;it!=m_activeObjectInds.end();++it) {
        AField * f = new AField;
        f->addVec3Channel("P", m_objects[*it]->numPoints());
        f->namedChannel("P")->copyFrom(m_objects[*it]->hostX(), m_objects[*it]->numPoints() * 12);
        m_positionFile->addField(objName(*it), f);
    }
}

bool CudaDynamicWorld::allFramesCached() const
{ return m_finishedCaching; }

std::string CudaDynamicWorld::objName(int i) const
{ 
    std::ostringstream oss;
    oss << "/massSystem_" << i;
    return oss.str();
}

void CudaDynamicWorld::updateSpeedLimit(float x)
{ m_contactSolver->setSpeedLimit(x + 20.f); }

float CudaDynamicWorld::totalEnergy() const
{ return m_totalEnergy; }

void CudaDynamicWorld::updateEnergy()
{
    m_totalEnergy = 0.f;
    for(unsigned i=0; i< numObjects(); i++) {
        m_totalEnergy += m_objects[i]->energy();
	}
}

void CudaDynamicWorld::putToSleep()
{
    for(unsigned i=0; i< numObjects(); i++) {
        if(m_objects[i]->isSleeping()) continue;
        if(m_objects[i]->velocitySize() < 2.9e-5f) {
            m_objects[i]->putToSleep();
        }
	}
}

bool CudaDynamicWorld::allSleeping() const
{
    for(unsigned i=0; i< numObjects(); i++) {
        if(!m_objects[i]->isSleeping()) return false;
    }
    return true;
}

void CudaDynamicWorld::wakeUpAll()
{
    for(unsigned i=0; i< numObjects(); i++) {
        m_objects[i]->wakeUp();
    }
}

void CudaDynamicWorld::resetAll()
{
    for(unsigned i=0; i< numObjects(); i++) {
        m_objects[i]->resetSystem();
    }
}

void CudaDynamicWorld::setPendingReset()
{ m_isPendingReset = true; }
//:~