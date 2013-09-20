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
	if(nf < 1) return;
	
	//setColor(1.f, 0.f, 0.f);
	
	unsigned i;
	for(i = 0; i < nf; i++) {
		MlCalamus * c = skin->getCalamus(i);
		drawAFeather(skin, c);
	}
	
	const unsigned num = skin->numActiveFeather();
	if(num < 1) return;
	
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
	
	
	Matrix33F ys;
	ys.rotateY(c->rotateY());
	ys.multiply(space);
	/*
	Vector3F d(0.f, 0.f, c->scale());
	d = ys.transform(d);
	d = p + d;
	arrow(p, d);*/
	
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