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

DrawNp::DrawNp() 
{
	m_x1 = new BaseBuffer;
	m_separateAxis = new BaseBuffer;
	m_localA = new BaseBuffer;
	m_localB = new BaseBuffer;
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
    computeX1(tetra);
    Vector3F * ptet = (Vector3F *)m_x1->data();
    
	m_separateAxis->create(phase->numContacts() * 16);
	m_localA->create(phase->numContacts() * 12);
	m_localB->create(phase->numContacts() * 12);
	
	phase->getSeparateAxis(m_separateAxis);
	phase->getLocalA(m_localA);
	phase->getLocalB(m_localB);
	
	unsigned * pairInd = (unsigned *)pairs->data();
	unsigned * tetInd = (unsigned *)tetra->hostTretradhedronIndices();
	
	float * sa = (float *)m_separateAxis->data();
	Vector3F * pa = (Vector3F *)m_localA->data();
	Vector3F * pb = (Vector3F *)m_localB->data();
	unsigned i;
	glColor3f(0.2f, 0.01f, 0.f);
	Vector3F dst, cenA, cenB;
	for(i=0; i < phase->numContacts(); i++) {
	    if(sa[i*4+3] < .1f) continue; 
	    cenA = tetrahedronCenter(ptet, tetInd, pairInd[i * 2]);
	    cenB = tetrahedronCenter(ptet, tetInd, pairInd[i * 2 + 1]);
		dst.set(sa[i*4], sa[i*4+1], sa[i*4+2]);
		m_drawer->arrow(cenB, cenB + dst);
		
		m_drawer->arrow(cenA, cenA + pa[i]);
		m_drawer->arrow(cenB, cenB + pb[i]);
	}
	
}

void DrawNp::computeX1(TetrahedronSystem * tetra)
{
    m_x1->create(tetra->numPoints() * 12);
	float * x1 = (float *)m_x1->data();
	
	float * x0 = tetra->hostX();
	float * vel = tetra->hostV();
	
	const float nf = tetra->numPoints() * 3;
	unsigned i;
	for(i=0; i < nf; i++)
		x1[i] = x0[i] + vel[i] * 0.01667f;
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

