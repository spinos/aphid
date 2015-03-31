/*
 *  DrawNp.cpp
 *  testnarrowpahse
 *
 *  Created by jian zhang on 3/3/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawNp.h"
#include <GeoDrawer.h>
#include <TetrahedronSystem.h>
#include <BaseBuffer.h>
#include <CudaNarrowphase.h>
#include <SimpleContactSolver.h>
#include "narrowphase_implement.h"
#include <CUDABuffer.h>
#include "radixsort_implement.h"
#include "simpleContactSolver_implement.h"
#include <CudaBase.h>
#include <BaseLog.h>
#include <stripedModel.h>
#include <boost/format.hpp>

DrawNp::DrawNp() 
{
	m_x1 = new BaseBuffer;
	m_coord = new BaseBuffer;
	m_contact = new BaseBuffer;
	m_counts = new BaseBuffer;
	m_contactPairs = new BaseBuffer;
	m_scanResult = new BaseBuffer;
	m_pairsHash = new BaseBuffer;
	m_linearVelocity = new BaseBuffer;
	m_angularVelocity = new BaseBuffer;
	m_deltaJ = new BaseBuffer;
	m_pntTetHash = new BaseBuffer;
	m_constraint = new BaseBuffer;
	m_tetPnt = new BaseBuffer;
	m_tetInd = new BaseBuffer;
	m_pointStarts = new BaseBuffer;
	m_indexStarts = new BaseBuffer;

	m_log = new BaseLog("narrowphase.txt");
}

DrawNp::~DrawNp() {}

void DrawNp::setDrawer(GeoDrawer * drawer)
{ m_drawer = drawer; }

void DrawNp::drawTetra(TetrahedronSystem * tetra)
{
	glColor3f(0.3f, 0.4f, 0.3f);
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
    glEnableClientState(GL_VERTEX_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)tetra->hostX());
	glDrawElements(GL_TRIANGLES, tetra->numTriangleFaceVertices(), GL_UNSIGNED_INT, tetra->hostTriangleIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
	
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	glColor3f(0.1f, 0.4f, 0.f);
	
    glEnableClientState(GL_VERTEX_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)tetra->hostX());
	glDrawElements(GL_TRIANGLES, tetra->numTriangleFaceVertices(), GL_UNSIGNED_INT, tetra->hostTriangleIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
}

void DrawNp::drawTetraAtFrameEnd(TetrahedronSystem * tetra)
{
	computeX1(tetra);
		
	glColor3f(0.21f, 0.21f, 0.24f);
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
    glEnableClientState(GL_VERTEX_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)m_x1->data());
	glDrawElements(GL_TRIANGLES, tetra->numTriangleFaceVertices(), GL_UNSIGNED_INT, tetra->hostTriangleIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
}

void DrawNp::drawSeparateAxis(CudaNarrowphase * phase, BaseBuffer * pairs, TetrahedronSystem * tetra)
{
    computeX1(tetra, 0.f);
    Vector3F * ptet = (Vector3F *)m_x1->data();
    
    const unsigned np = phase->numPairs();
	m_contact->create(np * 48);
	
	phase->getContact0(m_contact);
	
	unsigned * pairInd = (unsigned *)pairs->data();
	unsigned * tetInd = (unsigned *)tetra->hostTretradhedronIndices();
	
	ContactData * contact = (ContactData *)m_contact->data();
	glDisable(GL_DEPTH_TEST);
	unsigned i;
	
	Vector3F dst, cenA, cenB;
	for(i=0; i < np; i++) {
		ContactData & cf = contact[i];
		
	    cenA = tetrahedronCenter(ptet, tetInd, pairInd[i * 2]);
	    cenB = tetrahedronCenter(ptet, tetInd, pairInd[i * 2 + 1]);
	    
	    if(cf.separateAxis.w < 1.f) {
		    // std::cout<<"\n  penetrated!\n";
		    glColor3f(0.9f, 0.5f, 0.9f);
		    m_drawer->arrow(cenA, cenB);
	    
		    continue;
		}
	    
	    glColor3f(0.1f, 0.2f, 0.06f);
	    m_drawer->arrow(cenA, cenA + tetrahedronVelocity(tetra, tetInd, pairInd[i * 2]));
	    m_drawer->arrow(cenB, cenB + tetrahedronVelocity(tetra, tetInd, pairInd[i * 2 + 1]));
		
		m_drawer->setColor(.5f, 0.f, 0.f);
		
		m_drawer->arrow(cenB + Vector3F(cf.localB.x, cf.localB.y, cf.localB.z), 
		cenB + Vector3F(cf.localB.x + cf.separateAxis.x, cf.localB.y + cf.separateAxis.y, cf.localB.z + cf.separateAxis.z));
		
		m_drawer->setColor(0.f, .5f, 0.f);
		m_drawer->arrow(cenA, cenA + Vector3F(cf.localA.x, cf.localA.y, cf.localA.z));
		m_drawer->arrow(cenB, cenB + Vector3F(cf.localB.x, cf.localB.y, cf.localB.z));
	}
}

bool DrawNp::checkConstraint(SimpleContactSolver * solver, CudaNarrowphase * phase, TetrahedronSystem * tetra)
{
    const unsigned nc = phase->numContacts();
    if(nc < 1) return 1;
    
    m_log->write(boost::str(boost::format("\ncheck begin num contact %1% \n") % nc ));
    
    glDisable(GL_DEPTH_TEST);
    
    m_constraint->create(nc * 64);
    solver->constraintBuf()->deviceToHost(m_constraint->data(), nc * 64);
	ContactConstraint * constraint = (ContactConstraint *)m_constraint->data();
    
    const unsigned njacobi = solver->numIterations();
	
	m_deltaJ->create(nc * njacobi * 4);
	solver->deltaJBuf()->deviceToHost(m_deltaJ->data(), m_deltaJ->bufferSize());
	float * dJ = (float *)m_deltaJ->data();
    
    computeX1(tetra, 0.f);
    Vector3F * ptet = (Vector3F *)m_x1->data();
    unsigned * tetInd = (unsigned *)tetra->hostTretradhedronIndices();
    
    m_contactPairs->create(nc * 8);
    phase->getContactPairs(m_contactPairs);
    
    unsigned * c = (unsigned *)m_contactPairs->data();
    unsigned i, j;
    glColor3f(0.4f, 0.9f, 0.6f);
	Vector3F dst, cenA, cenB;
	for(i=0; i < nc; i++) {
	    // cenA = tetrahedronCenter(ptet, tetInd, c[i*2]);
	    // cenB = tetrahedronCenter(ptet, tetInd, c[i*2+1]);
		// m_drawer->arrow(cenB, cenA);
	}
    
	CUDABuffer * bodyPair = solver->contactPairHashBuf();
	m_pairsHash->create(bodyPair->bufferSize());
	bodyPair->deviceToHost(m_pairsHash->data(), m_pairsHash->bufferSize());
	
	m_linearVelocity->create(nc * 2 * 12);
	solver->deltaLinearVelocityBuf()->deviceToHost(m_linearVelocity->data(), 
	    m_linearVelocity->bufferSize());
	Vector3F * linVel = (Vector3F *)m_linearVelocity->data();
	
	m_angularVelocity->create(nc * 2 * 12);
	solver->deltaAngularVelocityBuf()->deviceToHost(m_angularVelocity->data(), 
	    m_angularVelocity->bufferSize());
	Vector3F * angVel = (Vector3F *)m_angularVelocity->data();
	
	m_contact->create(nc * 48);
	phase->contactBuffer()->deviceToHost(m_contact->data(), m_contact->bufferSize());
	ContactData * contact = (ContactData *)m_contact->data();
	
	float maxSAL = -1.f;
	float minSAL = 1e8;
	float SAL;
	Vector3F N;
	for(i=0; i<nc; i++) {
	    ContactData & cd = contact[i];
	    if(cd.separateAxis.w < .1f) continue;
	    float4 sa = cd.separateAxis;
	    N.set(sa.x, sa.y, sa.z);
	    SAL = N.length();
	    if(SAL > maxSAL) maxSAL = SAL;
	    if(SAL < minSAL) minSAL = SAL;
	}
	// std::cout<<"\n min max SA length "<<minSAL<<" "<<maxSAL<<"\n";
	
	bool isA;
	unsigned iPairA, iBody, iPair;
	unsigned * bodyAndPair = (unsigned *)m_pairsHash->data();
	bool converged;
	for(i=0; i < nc * 2; i++) {

	    iBody = bodyAndPair[i*2];
	    iPair = bodyAndPair[i*2+1];
	    
	    iPairA = iPair * 2;
// left or right
        isA = (iBody == c[iPairA]);

	    cenA = tetrahedronCenter(ptet, tetInd, iBody);
 
	    //cenB = cenA + angVel[i];
	    
	    //glColor3f(0.1f, 0.7f, 0.3f);
	    //m_drawer->arrow(cenA, cenB);
	    
	    ContactData & cd = contact[iPair];
	    float4 sa = cd.separateAxis;
	    N.set(sa.x, sa.y, sa.z);
	    N.reverse();
	    N.normalize();
	    
	    if(isA) {
// show contact normal for A
		    cenB = cenA + Vector3F(cd.localA.x, cd.localA.y, cd.localA.z);
		    m_drawer->setColor(0.f, .3f, .9f);
		    m_drawer->arrow(cenB, cenB + N);
		}

		glColor3f(0.73f, 0.68f, 0.1f);
		m_drawer->arrow(cenA, cenA + linVel[i]);
		
		glColor3f(0.1f, 0.68f, 0.72f);
		m_drawer->arrow(cenA, cenA + angVel[i]);
		
		if(isA) {
		    converged = 1;
		    m_log->write(boost::str(boost::format("\n contact[%1%]\n") % iPair));
			m_log->write(boost::str(boost::format(" toi %1%\n") % cd.timeOfImpact));
            
		    BarycentricCoordinate coord = constraint[iPair].coordA;
			float csum = coord.x + coord.y + coord.z + coord.w;
		    if(csum > 1.1f) converged = 0;
		    if(csum < .9f) converged = 0;
			
		    m_log->write(boost::str(boost::format(" coordA (%1%, %2%, %3%, %4%) sum %5%\n") % coord.x % coord.y % coord.z % coord.w % csum ));
		    coord = constraint[iPair].coordB;
		    csum = coord.x + coord.y + coord.z + coord.w;
		    if(csum > 1.1f) converged = 0;
		    if(csum < .9f) converged = 0;

		    m_log->write(boost::str(boost::format(" coordB (%1%, %2%, %3%, %4%) sum %5%\n") % coord.x % coord.y % coord.z % coord.w % csum ));
		    m_log->write(boost::str(boost::format(" relVel %1%\n") % constraint[iPair].relVel));
		    m_log->write(boost::str(boost::format(" Minv %1%\n") % constraint[iPair].Minv));
		    
		    m_log->write(boost::str(boost::format(" ra (%1%, %2%, %3%)\n") % cd.localA.x % cd.localA.y % cd.localA.z ));
		    m_log->write(boost::str(boost::format(" rb (%1%, %2%, %3%)\n") % cd.localB.x % cd.localB.y % cd.localB.z));

		    
		    if(constraint[iPair].relVel >0) converged = 0;
		    if(constraint[iPair].Minv >0) converged = 0;
			
		    m_log->write(boost::str(boost::format(" SA (%1%, %2%, %3%) length %4%\n") % sa.x % sa.y % sa.z % sqrt(sa.x * sa.x + sa.y * sa.y + sa.z * sa.z ) ));
		    m_log->write(boost::str(boost::format(" body[%1%]\n") % iBody));
		    
		    for(j=0; j< njacobi; j++) {
               m_log->write(boost::str(boost::format("dJ[%1%] %2%\n") % j % dJ[iPair * njacobi + j]));
            }
            
		    converged = 1;
            float lastJ = dJ[iPair * njacobi];
            if(lastJ < 1e-3) converged = 0;
			float sumJ = lastJ;
            for(j=1; j< njacobi; j++) {
                
                float curJ = dJ[iPair * njacobi + j];
                if(curJ < 0.f) curJ = -curJ;
                
                if(curJ >= lastJ && curJ > 1e-3) {
                    converged = 0;
                    break;
                }
                
                lastJ = curJ;
				sumJ+=curJ;
            }
			
			m_log->write(boost::str(boost::format("remain %1%\n") % (sumJ + constraint[iPair].relVel)));
            
            if(!converged) {
                std::cout<<" no converging!\n";
                m_log->write(" no converging!\n");
                std::cout<<" num contacts "<<nc<<"\n";
                glColor3f(1.f, 0.f, 0.f);
                m_drawer->arrow(cenA + Vector3F(0.f, -8.f, 0.f), cenA);
            }
        }
	}
	
	if(!converged) return 0;
	
	m_pntTetHash->create(nc * 2 * 4 * 8);
	solver->pntTetHashBuf()->deviceToHost(m_pntTetHash->data(), m_pntTetHash->bufferSize());
	KeyValuePair * pntHash = (KeyValuePair *)m_pntTetHash->data();
	
	// std::cout<<"pnt-tet hash\n";
	for(i=0; i < nc * 8; i++) {
	    // std::cout<<" pnt["<<i<<"] ("<<pntHash[i].key<<", "<<pntHash[i].value<<")\n";
	}
	
	m_log->write(boost::str(boost::format("device mem used %1% check end\n") % CudaBase::MemoryUsed ));
	return 1;
}

void DrawNp::showConstraint(SimpleContactSolver * solver, CudaNarrowphase * phase)
{
	const unsigned nc = phase->numContacts();
    if(nc < 1) return;
    
	CUDABuffer * pnts = phase->objectBuffer()->m_pos;
	CUDABuffer * inds = phase->objectBuffer()->m_ind;
	CUDABuffer * pointStarts = phase->objectBuffer()->m_pointCacheLoc;
	CUDABuffer * indexStarts = phase->objectBuffer()->m_indexCacheLoc;
	
	m_tetPnt->create(pnts->bufferSize());
	m_tetInd->create(inds->bufferSize());
	m_pointStarts->create(pointStarts->bufferSize());
	m_indexStarts->create(indexStarts->bufferSize());
	
	pnts->deviceToHost(m_tetPnt->data(), m_tetPnt->bufferSize());
	inds->deviceToHost(m_tetInd->data(), m_tetInd->bufferSize());
	pointStarts->deviceToHost(m_pointStarts->data(), m_pointStarts->bufferSize());
	indexStarts->deviceToHost(m_indexStarts->data(), m_indexStarts->bufferSize());
	
	Vector3F * tetPnt = (Vector3F *)m_tetPnt->data();
	unsigned * tetInd = (unsigned *)m_tetInd->data();
	unsigned * pntOffset = (unsigned *)m_pointStarts->data();
	unsigned * indOffset = (unsigned *)m_indexStarts->data();
	
    glDisable(GL_DEPTH_TEST);
    
    m_constraint->create(nc * 64);
    solver->constraintBuf()->deviceToHost(m_constraint->data(), nc * 64);
	ContactConstraint * constraint = (ContactConstraint *)m_constraint->data();
    
    m_contactPairs->create(nc * 8);
    phase->getContactPairs(m_contactPairs);
    
    unsigned * c = (unsigned *)m_contactPairs->data();
    unsigned i, j;
    glColor3f(0.4f, 0.9f, 0.6f);
	Vector3F dst, cenA, cenB;
    
	CUDABuffer * bodyPair = solver->contactPairHashBuf();
	m_pairsHash->create(bodyPair->bufferSize());
	bodyPair->deviceToHost(m_pairsHash->data(), m_pairsHash->bufferSize());
	
	m_linearVelocity->create(nc * 2 * 12);
	solver->deltaLinearVelocityBuf()->deviceToHost(m_linearVelocity->data(), 
	    m_linearVelocity->bufferSize());
	Vector3F * linVel = (Vector3F *)m_linearVelocity->data();
	
	m_angularVelocity->create(nc * 2 * 12);
	solver->deltaAngularVelocityBuf()->deviceToHost(m_angularVelocity->data(), 
	    m_angularVelocity->bufferSize());
	Vector3F * angVel = (Vector3F *)m_angularVelocity->data();
	
	m_contact->create(nc * 48);
	phase->contactBuffer()->deviceToHost(m_contact->data(), m_contact->bufferSize());
	ContactData * contact = (ContactData *)m_contact->data();

	Vector3F N;

	bool isA;
	unsigned iPairA, iBody, iPair;
	unsigned * bodyAndPair = (unsigned *)m_pairsHash->data();
	bool converged;
	for(i=0; i < nc * 2; i++) {

	    iBody = bodyAndPair[i*2];
	    iPair = bodyAndPair[i*2+1];
	    
	    // std::cout<<"body "<<iBody<<" pair "<<iPair<<"\n";
	    
	    iPairA = iPair * 2;
// left or right
        isA = (iBody == c[iPairA]);

	    cenA = tetrahedronCenter(tetPnt, tetInd, pntOffset, indOffset, iBody);
 
	    //cenB = cenA + angVel[i];
	    
	    //glColor3f(0.1f, 0.7f, 0.3f);
	    //m_drawer->arrow(cenA, cenB);
	    
	    ContactData & cd = contact[iPair];
	    float4 sa = cd.separateAxis;
	    N.set(sa.x, sa.y, sa.z);
	    N.reverse();
	    N.normalize();

	    if(isA) {
// show contact normal for A
		    cenB = cenA + Vector3F(cd.localA.x, cd.localA.y, cd.localA.z);
		    m_drawer->setColor(0.f, .3f, .9f);
		    m_drawer->arrow(cenB, cenB + N);
		}

		glColor3f(0.73f, 0.68f, 0.1f);
		m_drawer->arrow(cenA, cenA + linVel[i]);
		
		glColor3f(0.1f, 0.68f, 0.72f);
		m_drawer->arrow(cenA, cenA + angVel[i]);
		
		
	}
	
	glEnable(GL_DEPTH_TEST);
}

void DrawNp::printCoord(CudaNarrowphase * phase, BaseBuffer * pairs)
{
    const unsigned nc = phase->numContacts();
    m_coord->create(nc * 16);
    phase->getCoord(m_coord);
    float * coord = (float *)m_coord->data();
    unsigned i;
    for(i=0; i < nc; i++) {
        std::cout<<" "<<i<<":("<<coord[i*4]<<" "<<coord[i*4+1]<<" "<<coord[i*4+2]<<" "<<coord[i*4+3]<<") ";
    }
}

void DrawNp::printTOI(CudaNarrowphase * phase, BaseBuffer * pairs)
{
    const unsigned np = phase->numPairs();
    m_contact->create(np * 48);
	m_counts->create(np * 4);
    phase->getContact0(m_contact);
	phase->getContactCounts(m_counts);
    ContactData * contact = (ContactData *)m_contact->data();
	unsigned * counts = (unsigned *)m_counts->data();
    unsigned i;
    for(i=0; i < np; i++) {
        // if(counts[i])
		if(contact[i].timeOfImpact < .016667f) 
		std::cout<<" "<<i<<" "<<contact[i].timeOfImpact<<" ";
    }
	
	return;
	
	m_contactPairs->create(np * 8);
	m_scanResult->create(np * 4);
	
	if(phase->numContacts() < 1) return;
	
	phase->getContactPairs(m_contactPairs);
	phase->getScanResult(m_scanResult);
	
	unsigned * scans = (unsigned *)m_scanResult->data();
	
	for(i=0; i < np; i++) if(counts[i]) std::cout<<" i "<<i<<" to "<<scans[i]<<"\n";
	
	unsigned * squeezedPairs = (unsigned *)m_contactPairs->data();
	
	CudaNarrowphase::CombinedObjectBuffer * objectBuf = phase->objectBuffer();
	std::cout<<" n points "<<phase->numPoints();
	
	for(i=0; i < phase->numContacts(); i++) {
		std::cout<<" "<<i<<" ("<<squeezedPairs[i*2]<<", "<<squeezedPairs[i*2 + 1]<<")\n";
	}
}

void DrawNp::computeX1(TetrahedronSystem * tetra, float h)
{
    m_x1->create(tetra->numPoints() * 12);
	float * x1 = (float *)m_x1->data();
	
	float * x0 = tetra->hostX();
	float * vel = tetra->hostV();
	
	const float nf = tetra->numPoints() * 3;
	unsigned i;
	for(i=0; i < nf; i++)
		x1[i] = x0[i] + vel[i] * h;
}

Vector3F DrawNp::tetrahedronVelocity(TetrahedronSystem * tetra, unsigned * v, unsigned i)
{
    Vector3F * vel = (Vector3F *)tetra->hostV();
    Vector3F r = vel[v[i*4]];
    r += vel[v[i * 4 + 1]];
    r += vel[v[i * 4 + 2]];
    r += vel[v[i * 4 + 3]];
    r *= .25f;
    return r;
}

Vector3F DrawNp::tetrahedronCenter(Vector3F * p, unsigned * v, unsigned * pntOffset, unsigned * indOffset, unsigned i)
{
	unsigned objectI = extractObjectInd(i);
	unsigned elementI = extractElementInd(i);
	
	unsigned pStart = pntOffset[objectI];
	unsigned iStart = indOffset[objectI];
	
	Vector3F r = p[pStart + v[iStart + elementI * 4]];
    r += p[pStart + v[iStart + elementI * 4 + 1]];
    r += p[pStart + v[iStart + elementI * 4 + 2]];
    r += p[pStart + v[iStart + elementI * 4 + 3]];
    r *= .25f;
    
	return r;
}

Vector3F DrawNp::tetrahedronCenter(Vector3F * p, unsigned * v, unsigned i)
{
    Vector3F r = p[v[i * 4]];
    r += p[v[i * 4 + 1]];
    r += p[v[i * 4 + 2]];
    r += p[v[i * 4 + 3]];
    r *= .25f;
    return r;
}

Vector3F DrawNp::interpolatePointTetrahedron(Vector3F * p, unsigned * v, unsigned i, float * wei)
{
    Vector3F r = Vector3F::Zero;
    
    if(wei[0]> 1e-5) r += p[v[i * 4]] * wei[0];
    if(wei[1]> 1e-5) r += p[v[i * 4 + 1]] * wei[1];
    if(wei[2]> 1e-5) r += p[v[i * 4 + 2]] * wei[2];
    if(wei[3]> 1e-5) r += p[v[i * 4 + 3]] * wei[3];
    return r;
}

void DrawNp::printContactPairHash(SimpleContactSolver * solver, unsigned numContacts)
{
	if(numContacts < 1) return;
	CUDABuffer * hashp = solver->contactPairHashBuf();
	
	std::cout<<" nc "<<numContacts<<"\n";
	
	m_pairsHash->create(hashp->bufferSize());
	hashp->deviceToHost(m_pairsHash->data(), m_pairsHash->bufferSize());
	
	unsigned * v = (unsigned *)m_pairsHash->data();
	
	unsigned i;
	std::cout<<" body-contact hash ";
	for(i=0; i < numContacts * 2; i++) {
		std::cout<<" "<<i<<" ("<<v[i*2]<<","<<v[i*2+1]<<")\n";
	}
	
	hashp = solver->bodySplitLocBuf();
	m_pairsHash->create(hashp->bufferSize());
	hashp->deviceToHost(m_pairsHash->data(), m_pairsHash->bufferSize());
	
	std::cout<<" body-split pairs ";
	for(i=0; i < numContacts; i++) {
		std::cout<<" "<<i<<" ("<<v[i*2]<<","<<v[i*2+1]<<")\n";
	}
}
