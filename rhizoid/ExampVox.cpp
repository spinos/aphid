/*
 *  ExampVox.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ExampVox.h"
#include <KdEngine.h>
#include <VoxelGrid.h>

namespace aphid {

ExampVox::ExampVox() : 
m_boxCenterSizeF4(NULL),
m_boxPositionBuf(NULL),
m_boxNormalBuf(NULL),
m_numBoxes(0),
m_boxBufLength(0),
m_dopBufLength(0),
m_sizeMult(1.f)
{ 
	m_diffuseMaterialColV[0] = 0.47f;
	m_diffuseMaterialColV[1] = 0.46f;
	m_diffuseMaterialColV[2] = 0.45f;
	m_geomBox.setOne(); 
}

ExampVox::~ExampVox() 
{ 
	if(m_boxCenterSizeF4) delete[] m_boxCenterSizeF4; 
	if(m_boxPositionBuf) delete[] m_boxPositionBuf;
	if(m_boxNormalBuf) delete[] m_boxNormalBuf;
}

void ExampVox::voxelize1(sdb::VectorArray<cvx::Triangle> * tri,
							const BoundingBox & bbox)
{
	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 64;
	KdEngine engine;
	KdNTree<cvx::Triangle, KdNode4 > gtr;
	engine.buildTree<cvx::Triangle, KdNode4, 4>(&gtr, tri, bbox, &bf);
	
	VoxelGrid<cvx::Triangle, KdNode4>::BuildProfile vf;
	vf._maxLevel = 3;
	vf._extractDOP = true;
	vf._shellOnly = true;
	
	VoxelGrid<cvx::Triangle, KdNode4> ggd;
	ggd.create(&gtr, bbox, &vf);
	
	buildDOPDrawBuf(ggd.dops() );
}

void ExampVox::voxelize(const std::vector<Geometry *> & geoms)
{
	m_geomBox.reset();
	unsigned n = 0;
	std::vector<Geometry *>::const_iterator it = geoms.begin();
	for(;it!=geoms.end();++it) {
		n += (*it)->numComponents();
		m_geomBox.expandBy((*it)->calculateBBox() );
	}
	
	if(n < 1) return;
	
	m_geomBox.expand(0.01f);
	sdb::WorldGrid<GroupCell, unsigned > grid;
	grid.setGridSize(m_geomBox.getLongestDistance() / 15.f);
	
	it = geoms.begin();
	for(;it!=geoms.end();++it) {
		fillGrid(&grid, *it);
	}
	
	setNumBoxes(grid.size() );
	
	unsigned i=0;
	grid.begin();
	while(!grid.end()) {
		Vector3F center = grid.value()->m_box.center();
		m_boxCenterSizeF4[i*4] = center.x;
		m_boxCenterSizeF4[i*4+1] = center.y;
		m_boxCenterSizeF4[i*4+2] = center.z;
		m_boxCenterSizeF4[i*4+3] = grid.value()->m_box.getLongestDistance() * .67f;
	    i++;
		grid.next();   
	}
	
	buildBoxDrawBuf();
}

void ExampVox::fillGrid(sdb::WorldGrid<GroupCell, unsigned > * grid,
				Geometry * geo)
{
	const int n = geo->numComponents();
	for(int i = 0; i < n; i++) {
		BoundingBox ab = geo->calculateBBox(i);
		const Vector3F center = ab.center();
		GroupCell * c = grid->insertChild((const float *)&center);
		
		if(!c) {
			std::cout<<"\n error cast to GroupCell";
			return;
		}
		
		c->insert(i);
		c->m_box.expandBy(ab);
	}
}

void ExampVox::buildBoxDrawBuf() 
{
	for (unsigned i=0; i<m_numBoxes;++i) {
		setSolidBoxDrawBuffer(&m_boxCenterSizeF4[i*4], m_boxCenterSizeF4[i*4+3],
							&m_boxPositionBuf[i*36],
							&m_boxNormalBuf[i*36]);
	}
}

const BoundingBox & ExampVox::geomBox() const
{ return m_geomBox; }

const float & ExampVox::geomExtent() const
{ return m_geomExtent; }

const float & ExampVox::geomSize() const
{ return m_geomSize; }

const float * ExampVox::geomCenterV() const
{ return (const float *)&m_geomCenter; }

const Vector3F & ExampVox::geomCenter() const
{ return m_geomCenter; }

const float * ExampVox::geomScale() const
{ return m_geomScale; }

void ExampVox::drawGrid()
{ drawSolidBoxArray((const float *)m_boxPositionBuf, (const float *)m_boxNormalBuf, m_numBoxes * 36); }

void ExampVox::drawWireGrid()
{ drawWireBoxArray(m_boxCenterSizeF4, m_numBoxes); }

float * ExampVox::diffuseMaterialColV()
{ return m_diffuseMaterialColV; }

const unsigned & ExampVox::numBoxes() const
{ return m_numBoxes; }

float * ExampVox::boxCenterSizeF4()
{ return m_boxCenterSizeF4; }

bool ExampVox::setNumBoxes(unsigned n)
{
	if(n<1) return false;
	if(n<=m_numBoxes) {
		m_numBoxes = n;
		m_boxBufLength = n * 36;
		return false;
	}
	m_numBoxes = n;
	m_boxBufLength = n * 36;
	if(m_boxCenterSizeF4) delete[] m_boxCenterSizeF4;
	m_boxCenterSizeF4 = new float[n * 4];
	
	if(m_boxNormalBuf) delete[] m_boxNormalBuf;
	m_boxNormalBuf = new Vector3F[m_numBoxes * 36];
	
	if(m_boxPositionBuf) delete[] m_boxPositionBuf;
	m_boxPositionBuf = new Vector3F[m_numBoxes * 36];
	
	return true;
}

void ExampVox::setGeomSizeMult(const float & x)
{ m_sizeMult = x; }

void ExampVox::setGeomBox(const float & a, 
					const float & b,
					const float & c,
					const float & d,
					const float & e,
					const float & f)
{
	m_geomBox.m_data[0] = a;
	m_geomBox.m_data[1] = b;
	m_geomBox.m_data[2] = c;
	m_geomBox.m_data[3] = d;
	m_geomBox.m_data[4] = e;
	m_geomBox.m_data[5] = f;
	m_geomExtent = m_geomBox.radius();
	m_geomSize = m_sizeMult * sqrt((m_geomBox.distance(0) * m_geomBox.distance(2) ) / 6.f); 
	m_geomCenter.x = (a + d) * .5f;
	m_geomCenter.y = (b + e) * .5f;
	m_geomCenter.z = (c + f) * .5f;
	m_geomScale[0] = (d - a);
	m_geomScale[1] = (e - b);
	m_geomScale[2] = (f - c);
}

const float * ExampVox::diffuseMaterialColor() const
{ return m_diffuseMaterialColV; }

const float * ExampVox::boxCenterSizeF4() const
{ return m_boxCenterSizeF4; }

const float * ExampVox::boxNormalBuf() const
{ return (const float *)m_boxNormalBuf; }

const float * ExampVox::boxPositionBuf() const
{ return (const float *)m_boxPositionBuf; }

const unsigned & ExampVox::boxBufLength() const
{ return m_boxBufLength; }

void ExampVox::buildDOPDrawBuf(const sdb::VectorArray<AOrientedBox> & dops)
{
	const int ndop = dops.size();
	m_dopNormalBuf.reset(new Vector3F[ndop * 84]);
	m_dopPositionBuf.reset(new Vector3F[ndop * 84]);
	
	int bufLen = 0;
	DOP8Builder bud;
	int i=0, j;
	for(;i<ndop;++i) {
		bud.build(*dops.get(i) );
	
		for(j=0; j<bud.numTriangles() * 3; ++j) {
			m_dopNormalBuf.get()[bufLen + j] = bud.normal()[j];
			m_dopPositionBuf.get()[bufLen + j] = bud.vertex()[j];
		}
		
		bufLen += bud.numTriangles() * 3;
	}
	m_dopBufLength = bufLen;
}

const int & ExampVox::dopBufLength() const
{ return m_dopBufLength; }

void ExampVox::drawDop()
{
	if(m_dopBufLength < 1) return;
	drawSolidBoxArray((const float *)m_dopPositionBuf.get(),
						(const float *)m_dopNormalBuf.get(),
						m_dopBufLength);
}

}
