/*
 *  MlDrawer.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/15/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlDrawer.h"
#include "MlCalamus.h"
#include "MlSkin.h"
#include "MlTessellate.h"
#include <AccPatchMesh.h>
#include <PointInsidePolygonTest.h>
MlDrawer::MlDrawer()
{
	m_featherTess = new MlTessellate; 
}

MlDrawer::~MlDrawer() 
{
	delete m_featherTess;
}

void MlDrawer::drawFeather(MlSkin * skin) const
{
	const unsigned nf = skin->numFeathers();
	if(nf > 0) drawBuffer();
	
	const unsigned num = skin->numActiveFeather();
	if(num < 1) return;
	
	unsigned i;
	for(i = 0; i < num; i++) {
		MlCalamus * c = skin->getActive(i);
		drawAFeather(skin, c);
	}
}

void MlDrawer::drawAFeather(MlSkin * skin, MlCalamus * c) const
{
	Vector3F p;
	skin->getPointOnBody(c, p);
	Matrix33F frm = skin->tangentFrame(c);
	
	Matrix33F space;
	space.rotateX(c->rotateX());
	
	space.multiply(frm);
	
	c->collideWith(skin);
	c->computeFeatherWorldP(p, space);
	m_featherTess->setFeather(c->feather());
	m_featherTess->evaluate(c->feather());
	
	glEnableClientState( GL_VERTEX_ARRAY );
	glVertexPointer( 3, GL_FLOAT, 0, m_featherTess->vertices() );
	
	glEnableClientState( GL_NORMAL_ARRAY );
	glNormalPointer( GL_FLOAT, 0, m_featherTess->normals() );
	
	glDrawElements( GL_QUADS, m_featherTess->numIndices(), GL_UNSIGNED_INT, m_featherTess->indices());
	
	glDisableClientState( GL_NORMAL_ARRAY );
	glDisableClientState( GL_VERTEX_ARRAY );
}

void MlDrawer::rebuildBuffer(MlSkin * skin)
{
    const unsigned nf = skin->numFeathers();
	if(nf < 1) return;
	
	unsigned nv = 0, ni = 0;
    unsigned i, j;
	for(i = 0; i < nf; i++) {
		MlCalamus * c = skin->getCalamus(i);
		m_featherTess->setFeather(c->feather());
		nv += m_featherTess->numVertices();
		ni += m_featherTess->numIndices();
	}
	
	createBuffer(nv, ni);
	
	unsigned curv = 0, curi = 0, nvpf, nipf;
	for(i = 0; i < nf; i++) {
		MlCalamus * c = skin->getCalamus(i);
		Vector3F p;
		skin->getPointOnBody(c, p);
		Matrix33F frm = skin->tangentFrame(c);
	
		Matrix33F space;
		space.rotateX(c->rotateX());
	
		space.multiply(frm);
	
		c->collideWith(skin);
		c->computeFeatherWorldP(p, space);
		m_featherTess->setFeather(c->feather());
		m_featherTess->evaluate(c->feather());
		
		nvpf = m_featherTess->numVertices();
		for(j = 0; j < nvpf; j++) {
		    m_vertices[curv + j] = m_featherTess->vertices()[j];
		    m_normals[curv + j] = m_featherTess->normals()[j];
		}
		
		nipf = m_featherTess->numIndices();
		for(j = 0; j < nipf; j++) {
		    m_indices[curi + j] = curv + m_featherTess->indices()[j];
		}
		
		curv += nvpf;
		curi += nipf;
	}
	
	std::cout<<"nv "<< curv;
	std::cout<<"ni "<< curi;
}