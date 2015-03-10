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
#include "narrowphase_implement.h"

DrawNp::DrawNp() 
{
	m_x1 = new BaseBuffer;
	m_coord = new BaseBuffer;
	m_contact = new BaseBuffer;
	m_counts = new BaseBuffer;
	m_contactPairs = new BaseBuffer;
	m_scanResult = new BaseBuffer;
}

DrawNp::~DrawNp() {}

void DrawNp::setDrawer(GeoDrawer * drawer)
{ m_drawer = drawer; }

void DrawNp::drawTetra(TetrahedronSystem * tetra)
{
	glColor3f(0.1f, 0.4f, 0.3f);
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
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
	
	unsigned i;
	glColor3f(0.2f, 0.01f, 0.f);
	Vector3F dst, cenA, cenB;
	for(i=0; i < np; i++) {
		ContactData & cf = contact[i];
		
		if(cf.separateAxis.w < .1f) continue;
	    cenA = tetrahedronCenter(ptet, tetInd, pairInd[i * 2]);
	    cenB = tetrahedronCenter(ptet, tetInd, pairInd[i * 2 + 1]);
		
		m_drawer->setColor(.5f, 0.f, 0.f);
		
		m_drawer->arrow(cenB + Vector3F(cf.localB.x, cf.localB.y, cf.localB.z), 
		cenB + Vector3F(cf.localB.x + cf.separateAxis.x, cf.localB.y + cf.separateAxis.y, cf.localB.z + cf.separateAxis.z));
		
		m_drawer->setColor(0.f, .5f, 0.f);
		m_drawer->arrow(cenA, cenA + Vector3F(cf.localA.x, cf.localA.y, cf.localA.z));
		m_drawer->arrow(cenB, cenB + Vector3F(cf.localB.x, cf.localB.y, cf.localB.z));
	}
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

