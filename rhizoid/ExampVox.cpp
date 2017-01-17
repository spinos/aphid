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
#include <ConvexShape.h>
#include <sdb/VectorArray.h>

namespace aphid {

ExampVox::ExampVox() : 
m_sizeMult(1.f)
{ 
    memset(m_defInstance._trans, 0, 64);
    m_defInstance._trans[0] = 1.f;
    m_defInstance._trans[5] = 1.f;
    m_defInstance._trans[10] = 1.f;
    m_defInstance._trans[15] = 1.f;
    m_defInstance._exampleId = 0;
    m_defInstance._instanceId = 0;
	m_diffuseMaterialColV[0] = 0.47f;
	m_diffuseMaterialColV[1] = 0.46f;
	m_diffuseMaterialColV[2] = 0.48f;
	m_dopSize.set(.9f, .9f, .9f);
	m_geomBox.setOne(); 
}

ExampVox::~ExampVox() 
{}

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
	buildBounding8Dop(bbox);
}

void ExampVox::buildBounding8Dop(const BoundingBox & bbox)
{
	AOrientedBox ob;
	//Matrix33F zup;
	//zup.rotateX(-1.57f);
	//ob.setOrientation(zup);
	ob.caluclateOrientation(&bbox);
	ob.calculateCenterExtents(triPositionBuf(), triBufLength(), &bbox );
	update8DopPoints(ob, (const float * )&m_dopSize);
}

const BoundingBox & ExampVox::geomBox() const
{ return m_geomBox; }

const float & ExampVox::geomExtent() const
{ return m_geomExtent; }

const float & ExampVox::geomSize() const
{ return m_geomSize; }

//const float * ExampVox::geomCenterV() const
//{ return (const float *)&m_geomCenter; }

const Vector3F & ExampVox::geomCenter() const
{ return m_geomCenter; }

float * ExampVox::diffuseMaterialColV()
{ return m_diffuseMaterialColV; }

void ExampVox::setGeomSizeMult(const float & x)
{ m_sizeMult = x; }

void ExampVox::setGeomBox(BoundingBox * bx)
{
/// limit box (-16 0 -16 16 16 16)
	if(bx->distance(0) < 1e-1f) {
		bx->setMin(-16.f, 0);
		bx->setMax( 16.f, 0);
	}
	if(bx->distance(1) < 1e-1f) {
		bx->setMin( 0.f, 1);
		bx->setMax( 16.f, 1);
	}
	if(bx->distance(2) < 1e-1f) {
		bx->setMin(-16.f, 2);
		bx->setMax( 16.f, 2);
	}
	m_geomBox = *bx;
	m_geomExtent = m_geomBox.radius();
	m_geomSize = m_sizeMult * sqrt((m_geomBox.distance(0) * m_geomBox.distance(2) ) / 6.f); 
	m_geomCenter = m_geomBox.center();
	updatePoints(&m_geomBox);
}

const float * ExampVox::diffuseMaterialColor() const
{ return m_diffuseMaterialColV; }

void ExampVox::drawWiredBound() const
{
	drawBoundingBox(&m_geomBox);
}

void ExampVox::drawSolidBound() const
{
	drawASolidBox();
}

void ExampVox::setDopSize(const float & a,
	                const float & b,
	                const float &c)
{
    m_dopSize.set(a, b, c);
}

const float * ExampVox::dopSize() const
{ return (const float * )&m_dopSize; }

int ExampVox::numExamples() const
{ return 1; }

int ExampVox::numInstances() const
{ return 1; }

const ExampVox * ExampVox::getExample(const int & i) const
{ return this; } 

ExampVox * ExampVox::getExample(const int & i)
{ return this; }

const ExampVox::InstanceD & ExampVox::getInstance(const int & i) const
{ return m_defInstance; }

}
