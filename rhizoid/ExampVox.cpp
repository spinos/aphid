/*
 *  ExampVox.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ExampVox.h"
#include <UniformGrid.h>
#include <KdIntersection.h>

ExampVox::ExampVox() : 
m_boxCenterSizeF4(NULL),
m_numBoxes(0)
{ m_geomBox.setOne(); }

ExampVox::~ExampVox() 
{ if(m_boxCenterSizeF4) delete[] m_boxCenterSizeF4; }

void ExampVox::voxelize(KdIntersection * tree)
{ 
	m_geomBox = tree->getBBox();
	m_geomBox.expand(0.01f);
	UniformGrid grd;
	grd.setBounding(m_geomBox);
	grd.create(tree, 4);
	unsigned n = grd.numCells();
	if(n < 1) return;
	
	setNumBoxes(n);
	
	if(m_boxCenterSizeF4) delete[] m_boxCenterSizeF4;
	m_boxCenterSizeF4 = new float[m_numBoxes * 4];
	
	unsigned i=0;
	sdb::CellHash * c = grd.cells();
	c->begin();
	while(!c->end()) {
		Vector3F center = grd.cellCenter(c->key() );
		m_boxCenterSizeF4[i*4] = center.x;
		m_boxCenterSizeF4[i*4+1] = center.y;
		m_boxCenterSizeF4[i*4+2] = center.z;
		m_boxCenterSizeF4[i*4+3] = grd.cellSizeAtLevel(c->value()->level);
	    i++;
		c->next();   
	}
}

const BoundingBox & ExampVox::geomBox() const
{ return m_geomBox; }

void ExampVox::setGeomBox(const BoundingBox & x)
{ m_geomBox = x; }

void ExampVox::drawGrid()
{
	unsigned i=0;
	for(;i<m_numBoxes;++i)
	    drawSolidBox((const float *)&m_boxCenterSizeF4[i*4], m_boxCenterSizeF4[i*4+3] );
}

void ExampVox::drawWireGrid()
{
	unsigned i=0;
	for(;i<m_numBoxes;++i)
	    drawWireBox((const float *)&m_boxCenterSizeF4[i*4], m_boxCenterSizeF4[i*4+3] );
}

float * ExampVox::diffuseMaterialColV()
{ return m_diffuseMaterialColV; }

const unsigned & ExampVox::numBoxes() const
{ return m_numBoxes; }

float * ExampVox::boxCenterSizeF4()
{ return m_boxCenterSizeF4; }

void ExampVox::setNumBoxes(unsigned n)
{
	m_numBoxes = n;
	if(m_boxCenterSizeF4) delete[] m_boxCenterSizeF4;
	m_boxCenterSizeF4 = new float[n * 4];
}
