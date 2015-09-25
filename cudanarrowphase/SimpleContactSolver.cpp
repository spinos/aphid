/*
 *  SimpleContactSolver.cpp
 *  testnarrowpahse
 *
 *  Created by jian zhang on 3/8/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "SimpleContactSolver.h"
#include "CudaNarrowphase.h"
#include <CUDABuffer.h>
#include "simpleContactSolver_implement.h"
#include <DynGlobal.h>
#include <CudaBase.h>

#if 1
#include <CudaDbgLog.h>
CudaDbgLog svlg("solver.txt");
#include <vector>
#include <utility>
std::vector<std::pair<int, int> > constraintDesc;


#endif

SimpleContactSolver::SimpleContactSolver() 
{
#if 1
    constraintDesc.push_back(std::pair<int, int>(1, 0));
    constraintDesc.push_back(std::pair<int, int>(1, 4));
    constraintDesc.push_back(std::pair<int, int>(1, 8));
    constraintDesc.push_back(std::pair<int, int>(1, 12));
    constraintDesc.push_back(std::pair<int, int>(1, 16));
    constraintDesc.push_back(std::pair<int, int>(1, 20));
    constraintDesc.push_back(std::pair<int, int>(1, 24));
    constraintDesc.push_back(std::pair<int, int>(1, 28));
#endif
	m_sortedInd[0] = new CUDABuffer;
	m_sortedInd[1] = new CUDABuffer;
	m_splitPair = new CUDABuffer;
	m_bodyCount = new CUDABuffer;
	m_splitInverseMass = new CUDABuffer;
	m_constraint = new CUDABuffer;
	m_deltaLinearVelocity = new CUDABuffer;
	m_deltaAngularVelocity = new CUDABuffer;
	m_contactLinearVelocity = new CUDABuffer;
	m_relVel = new CUDABuffer;
	m_pntTetHash[0] = new CUDABuffer;
	m_pntTetHash[1] = new CUDABuffer;
	m_bodyTetInd = new CUDABuffer;
	m_numContacts = 0;
	// std::cout<<" sizeof struct ContactConstraint "<<sizeof(ContactConstraint)<<"\n";
}

SimpleContactSolver::~SimpleContactSolver() {}

void SimpleContactSolver::initOnDevice()
{
	CudaReduction::initOnDevice();
}

CUDABuffer * SimpleContactSolver::contactPairHashBuf()
{ return m_sortedInd[0]; }

CUDABuffer * SimpleContactSolver::bodySplitLocBuf()
{ return m_splitPair; }

CUDABuffer * SimpleContactSolver::constraintBuf()
{ return m_constraint; }

CUDABuffer * SimpleContactSolver::deltaLinearVelocityBuf()
{ return m_deltaLinearVelocity; }

CUDABuffer * SimpleContactSolver::deltaAngularVelocityBuf()
{ return m_deltaAngularVelocity; }

CUDABuffer * SimpleContactSolver::pntTetHashBuf()
{ return m_pntTetHash[0]; }

CUDABuffer * SimpleContactSolver::splitInverseMassBuf()
{ return m_splitInverseMass; }

const unsigned SimpleContactSolver::numContacts() const
{ return m_numContacts; }

void SimpleContactSolver::solveContacts(unsigned numContacts,
										CUDABuffer * contactBuf,
										CUDABuffer * pairBuf,
										void * objectData)
{
#if DISABLE_COLLISION_RESOLUTION
	return;
#endif
    if(numContacts < 1) return; 
    
#if 0
    svlg.writeInt2( pairBuf,
                    numContacts,
                   "pair", CudaDbgLog::FAlways);
#endif
    
	const unsigned indBufLength = iRound1024(numContacts * 2);
	
	m_sortedInd[0]->create(indBufLength * 8);	
	m_sortedInd[1]->create(indBufLength * 8);
	
	void * bodyContactHash = m_sortedInd[0]->bufferOnDevice();
	void * pairs = pairBuf->bufferOnDevice();
	
/*  
 *  for either side of each contact pair, set
 *  key: body index
 *  velue: contact index
 *  n x 2 hash
 *  sort by body index to put the same body together 
 */
	simpleContactSolverWriteContactIndex((KeyValuePair *)bodyContactHash, 
	    (uint *)pairs, 
	    numContacts * 2, 
	    indBufLength);
	
	void * tmp = m_sortedInd[1]->bufferOnDevice();
	RadixSort((KeyValuePair *)bodyContactHash, (KeyValuePair *)tmp, indBufLength, 30);

#if 0
    svlg.writeHash( m_sortedInd[0],
                    numContacts * 2,
                   "body-contact", CudaDbgLog::FAlways);
#endif

/*
 *  for each hash, find the index of contact pair
 *  set the indirection from contact pair to hash index
 */
	m_splitPair->create(numContacts * 8);
	void * splits = m_splitPair->bufferOnDevice();
	
	const unsigned splitBufLength = numContacts * 2;
	simpleContactSolverComputeSplitBufLoc((uint2 *)splits, 
	                        (uint2 *)pairs, 
	                        (KeyValuePair *)bodyContactHash, 
	                        splitBufLength);

#if 0
    svlg.writeInt2( m_splitPair,
                    numContacts,
                   "splitpair", CudaDbgLog::FAlways);
#endif
	
	m_bodyCount->create(splitBufLength * 4);
	void * bodyCount = m_bodyCount->bufferOnDevice();
	simpleContactSolverCountBody((uint *)bodyCount, 
	                        (KeyValuePair *)bodyContactHash, 
	                        splitBufLength);

#if 0
// num iterattions by max contacts per object
// todo ignore static object count
	int mxcount = 0;
	max<int>(mxcount, (int *)bodyCount, splitBufLength);
	int numiterations = mxcount + 3;
#else
	int numiterations = 9;
#endif
	
	m_splitInverseMass->create(splitBufLength * 4);
	void * splitMass = m_splitInverseMass->bufferOnDevice();
	
	CudaNarrowphase::CombinedObjectBuffer * objectBuf = (CudaNarrowphase::CombinedObjectBuffer *)objectData;
	void * pos = objectBuf->m_pos->bufferOnDevice();
	void * vel = objectBuf->m_vel->bufferOnDevice();
	void * mass = objectBuf->m_mass->bufferOnDevice();
    void * linearImpulse = objectBuf->m_linearImpulse->bufferOnDevice();
	void * ind = objectBuf->m_ind->bufferOnDevice();
	void * perObjPointStart = objectBuf->m_pointCacheLoc->bufferOnDevice();
	void * perObjectIndexStart = objectBuf->m_indexCacheLoc->bufferOnDevice();
	m_bodyTetInd->create(4* 4 * numContacts *2);
	
	simpleContactSolverComputeSplitInverseMass((float *)splitMass,
	                        (uint2 *)splits,
	                        (uint2 *)pairs,
	                        (float *)mass,
	                        (uint4 *)ind,
	                        (uint * )perObjPointStart,
	                        (uint * )perObjectIndexStart,
                            (uint *)bodyCount,
                            (uint4 *)m_bodyTetInd->bufferOnDevice(),
                            numContacts * 2);
    
#if 0
   // svlg.writeFlt( m_splitInverseMass,
     //               numContacts,
       //            "masstensor", CudaDbgLog::FAlways);
    
    svlg.writeUInt( objectBuf->m_pointCacheLoc,
                    2,
                   "pstart", CudaDbgLog::FAlways);
    svlg.writeUInt( objectBuf->m_indexCacheLoc,
                    2,
                   "istart", CudaDbgLog::FAlways);
#endif
	
    m_constraint->create(numContacts * 64);
	m_contactLinearVelocity->create(numContacts * 2 * 12);
	void * constraint = m_constraint->bufferOnDevice();
	void * contactLinearVel = m_contactLinearVelocity->bufferOnDevice();
	void * contacts = contactBuf->bufferOnDevice();
	
	contactconstraint::prepareNoPenetratingContact((ContactConstraint *)constraint,
	    (float3 *)contactLinearVel,
	    (uint2 *)splits,
	    (uint2 *)pairs,
	    (float3 *)pos,
	    (float3 *)vel,
        (float3 *)linearImpulse,
	    (float *)splitMass,
	    (ContactData *)contacts,
	    (uint4 *)m_bodyTetInd->bufferOnDevice(),
        numContacts * 2);
    CudaBase::CheckCudaError("jacobi solver prepare constraint");
#if 0
    svlg.writeUInt( m_bodyTetInd,
                    numContacts * 8,
                   "tet", CudaDbgLog::FAlways);
#endif
#if 0
    
    svlg.writeFlt( contactBuf,
                    numContacts * 12,
                   "contact", CudaDbgLog::FAlways);
#endif

#if 0
    svlg.writeStruct(m_constraint, numContacts, 
                   "constraint", 
                   constraintDesc,
                   64,
                   CudaDbgLog::FAlways);
   // svlg.writeVec3(m_contactLinearVelocity, numContacts * 2, 
     //              "contact_vel", CudaDbgLog::FAlways);
#endif

	m_deltaLinearVelocity->create(nextPow2(splitBufLength * 12));
	m_deltaAngularVelocity->create(nextPow2(splitBufLength * 12));
	
	void * deltaLinVel = m_deltaLinearVelocity->bufferOnDevice();
	void * deltaAngVel = m_deltaAngularVelocity->bufferOnDevice();
	simpleContactSolverClearDeltaVelocity((float3 *)deltaLinVel, 
	                            (float3 *)deltaAngVel, 
	                            splitBufLength);

	int i;
	for(i=0; i< numiterations; i++) {
// compute impulse and velocity changes per contact
        contactconstraint::resolveCollision((ContactConstraint *)constraint,
	                    (float3 *)contactLinearVel,
                        (float3 *)deltaLinVel,
	                    (uint2 *)pairs,
	                    (uint2 *)splits,
	                    (float *)splitMass,
	                    (ContactData *)contacts,
	                    numContacts * 2);
        CudaBase::CheckCudaError("jacobi solver resolve collision");
        
#if 0
    unsigned ii = i;
    svlg.write(ii);
#endif
#if 0
    svlg.writeVec3(m_deltaLinearVelocity, numContacts * 2, 
                   "deltaV_b4", CudaDbgLog::FAlways);
#endif
    
	    simpleContactSolverAverageVelocities((float3 *)deltaLinVel,
                        (float3 *)deltaAngVel,
                        (uint *)bodyCount,
                        (KeyValuePair *)bodyContactHash, 
                        splitBufLength);
        CudaBase::CheckCudaError("jacobi solver average velocity");
        
#if 0
    svlg.writeVec3(m_deltaLinearVelocity, numContacts * 2, 
                   "deltaV_avg", CudaDbgLog::FAlways);
#endif

        contactconstraint::resolveFriction((ContactConstraint *)constraint,
	                    (float3 *)contactLinearVel,
                        (float3 *)deltaLinVel,
	                    (uint2 *)pairs,
	                    (uint2 *)splits,
	                    (float *)splitMass,
	                    (ContactData *)contacts,
	                    numContacts * 2);
        CudaBase::CheckCudaError("jacobi solver resolve friction");
        
        simpleContactSolverAverageVelocities((float3 *)deltaLinVel,
                        (float3 *)deltaAngVel,
                        (uint *)bodyCount,
                        (KeyValuePair *)bodyContactHash, 
                        splitBufLength);
        CudaBase::CheckCudaError("jacobi solver average velocity");

	}

// 2 tet per contact, 4 pnt per tet, key is pnt index, value is tet index in split
	const unsigned pntHashBufLength = iRound1024(numContacts * 2 * 4);
    // std::cout<<"\n pntHashBufLength"<<pntHashBufLength
    // <<" numContact"<<numContacts;
	m_pntTetHash[0]->create(pntHashBufLength * 8);
	m_pntTetHash[1]->create(pntHashBufLength * 8);
	
	void * pntTetHash = m_pntTetHash[0]->bufferOnDevice();
	
	simpleContactSolverWritePointTetHash((KeyValuePair *)pntTetHash,
	                (uint2 *)pairs,
	                (uint2 *)splits,
	                (uint *)bodyCount,
	                (uint4 *)ind,
	                (uint * )perObjPointStart,
	                (uint * )perObjectIndexStart,
	                numContacts * 2,
	                pntHashBufLength);
    CudaBase::CheckCudaError(// CudaBase::Synchronize(),
                             "jacobi solver point-tetra hash");
    
	void * intermediate = m_pntTetHash[1]->bufferOnDevice();
	RadixSort((KeyValuePair *)pntTetHash, (KeyValuePair *)intermediate, pntHashBufLength, 24);

#if 0
    svlg.writeHash(m_pntTetHash[1], numContacts * 2, 
                   "pnttet_hash", CudaDbgLog::FAlways);
#endif
    
    contactsolver::updateImpulse((float3 *)linearImpulse,
	                (float3 *)deltaLinVel,
	                (float3 *)deltaAngVel,
	                (KeyValuePair *)pntTetHash,
                    (uint2 *)pairs,
                    (uint2 *)splits,
                    (ContactConstraint *)constraint,
                    (ContactData *)contacts,
                    (float3 *)pos,
                    (uint4 *)ind,
                    (uint * )perObjPointStart,
                    (uint * )perObjectIndexStart,
                    numContacts * 2 * 4);
    CudaBase::CheckCudaError(// CudaBase::Synchronize(),
        "jacobi solver update velocity");
}

void SimpleContactSolver::setSpeedLimit(float x)
{ contactsolver::setSpeedLimit(x); }
//:~