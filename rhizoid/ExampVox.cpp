/*
 *  ExampVox.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ExampVox.h"
#include <kd/KdEngine.h>
#include <FieldTriangulation.h>

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

void ExampVox::voxelize2(sdb::VectorArray<cvx::Triangle> * tri,
							const BoundingBox & bbox)
{
	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 64;
	KdEngine engine;
	KdNTree<cvx::Triangle, KdNode4 > gtr;
	engine.buildTree<cvx::Triangle, KdNode4, 4>(&gtr, tri, bbox, &bf);
	
	BoundingBox tb = gtr.getBBox();
	const float gz = tb.getLongestDistance() * 1.23f;
	const Vector3F cent = tb.center();
	tb.setMin(cent.x - gz, cent.y - gz, cent.z - gz );
	tb.setMax(cent.x + gz, cent.y + gz, cent.z + gz );
	
	ttg::FieldTriangulation msh;
	msh.fillBox(tb, gz);
	
	BDistanceFunction distFunc;
	distFunc.addTree(&gtr);
	distFunc.setDomainDistanceRange(gz * GDT_FAC_ONEOVER8 );
	
	msh.frontAdaptiveBuild<BDistanceFunction>(&distFunc, 3, 4, .3f);
	msh.triangulateFront();
	
	std::cout.flush();
	
	buildTriangleDrawBuf(msh.numFrontTriangles(), msh.triangleIndices(),
						msh.numVertices(), msh.triangleVertexP(), msh.triangleVertexN() );
						
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
		return true;
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

Vector3F * ExampVox::dopNormalR()
{ return m_dopNormalBuf.get(); }
	
Vector3F * ExampVox::dopPositionR()
{ return m_dopPositionBuf.get(); }

const float * ExampVox::dopNormalBuf() const
{ return (const float *)m_dopNormalBuf.get(); }

const float * ExampVox::dopPositionBuf() const
{ return (const float *)m_dopPositionBuf.get(); }

void ExampVox::setDOPDrawBufLen(const int & x)
{ 
	m_dopBufLength = x;
	m_dopNormalBuf.reset(new Vector3F[x]);
	m_dopPositionBuf.reset(new Vector3F[x]);
}

void ExampVox::buildTriangleDrawBuf(const int & nt, const int * tri,
				const int & nv, const Vector3F * vertP, const Vector3F * vertN )
{
	m_dopBufLength = nt * 3;
	m_dopNormalBuf.reset(new Vector3F[m_dopBufLength]);
	m_dopPositionBuf.reset(new Vector3F[m_dopBufLength]);
	
	int i=0, j;
	for(;i<m_dopBufLength;++i) {
		j = tri[i]; 
		m_dopNormalBuf.get()[i] = vertN[j];
		m_dopPositionBuf.get()[i] = vertP[j];
	}
	
}

}
