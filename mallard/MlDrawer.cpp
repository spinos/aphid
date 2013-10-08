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
	if(skin->numFeathers() > 0) drawBuffer();
}

void MlDrawer::hideAFeather(MlCalamus * c)
{
	const unsigned loc = c->bufferStart();
	
	setIndex(loc);
	
	m_featherTess->setFeather(c->feather());
	
	unsigned i;
	const unsigned nvpf = m_featherTess->numIndices();
	for(i = 0; i < nvpf; i++) {
		vertices()[0] = 0.f;
		vertices()[1] = 0.f;
		vertices()[2] = 0.f;
		next();
	}
}

void MlDrawer::hideActive(MlSkin * skin)
{
	const unsigned num = skin->numActive();
	if(num < 1) return;
	
	unsigned i;
	for(i = 0; i < num; i++) {
		MlCalamus * c = skin->getActive(i);
		hideAFeather(c);
	}
}

void MlDrawer::updateActive(MlSkin * skin)
{
	const unsigned num = skin->numActive();
	if(num < 1) return;
	
	unsigned i;
	for(i = 0; i < num; i++) {
		MlCalamus * c = skin->getActive(i);
		computeAFeather(skin, c);
	}
}

void MlDrawer::computeAFeather(MlSkin * skin, MlCalamus * c)
{
	tessellate(skin, c);
	
	const unsigned nvpf = m_featherTess->numIndices();
	const unsigned startv = c->bufferStart();
	setIndex(startv);
	
	unsigned i, j;
	Vector3F v;
	for(i = 0; i < nvpf; i++) {
		j = m_featherTess->indices()[i];
		v = m_featherTess->vertices()[j];
		vertices()[0] = v.x;
		vertices()[1] = v.y;
		vertices()[2] = v.z;
		
		v = m_featherTess->normals()[j];
		normals()[0] = v.x;
		normals()[1] = v.y;
		normals()[2] = v.z;
		
		next();
	}
}

void MlDrawer::addToBuffer(MlSkin * skin)
{
	const unsigned num = skin->numCreated();
	if(num < 1) return;
	
	unsigned loc = taken();
	
	unsigned i, nvpf;
	Vector3F v;
	for(i = 0; i < num; i++) {
		MlCalamus * c = skin->getCreated(i);
		
		m_featherTess->setFeather(c->feather());
		
		setIndex(loc);
		c->setBufferStart(loc);
		
		nvpf = m_featherTess->numIndices();

		expandBy(nvpf);
		
		loc += nvpf;
	}
}

void MlDrawer::rebuildBuffer(MlSkin * skin)
{
	unsigned nc = skin->numFeathers();
	if(nc < 1) return;
	
	//std::cout<<"building feather draw buffer\n num feathers "<<nc<<"\n";
	unsigned loc = 0;
	
	unsigned i, nvpf;
	Vector3F v;
	for(i = 0; i < nc; i++) {
		MlCalamus * c = skin->getCalamus(i);
		
		m_featherTess->setFeather(c->feather());
		
		setIndex(loc);
		c->setBufferStart(loc);
		
		nvpf = m_featherTess->numIndices();

		expandBy(nvpf);
		
		loc += nvpf;
	}
	
	for(i = 0; i < nc; i++) {
		MlCalamus * c = skin->getCalamus(i);
		computeAFeather(skin, c);
	}
}

void MlDrawer::tessellate(MlSkin * skin, MlCalamus * c)
{
	Vector3F p;
	skin->getPointOnBody(c, p);

	Matrix33F space = skin->rotationFrame(c);

	c->collideWith(skin, p);
	c->computeFeatherWorldP(p, space);
	m_featherTess->setFeather(c->feather());
	m_featherTess->evaluate(c->feather());
}
