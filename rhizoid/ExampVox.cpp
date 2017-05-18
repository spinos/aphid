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
//#include <FieldTriangulation.h>
#include <geom/ConvexShape.h>
#include <sdb/VectorArray.h>
#include <kd/IntersectEngine.h>
#include <kd/ClosestToPointEngine.h>
#include <sdb/LodSampleCache.h>
#include <topo/KHullGen.h>

namespace aphid {

ExampVox::ExampVox() : 
m_sizeMult(1.f),
m_isActive(true),
m_isVisible(true),
m_drawDetailType(0)
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
	m_dopSize.set(.9f, 1.f, .9f);
	m_geomBox.setOne(); 
}

ExampVox::~ExampVox() 
{}
/*
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
	tb.set(cent, gz );
	
	ttg::FieldTriangulation msh;
	msh.fillBox(tb, gz);
	
	BDistanceFunction distFunc;
	distFunc.addTree(&gtr);
	distFunc.setDomainDistanceRange(gz * GDT_FAC_ONEOVER8 );
	
	msh.frontAdaptiveBuild<BDistanceFunction>(&distFunc, 3, 4, .3f);
	msh.triangulateFront();
	
	std::cout.flush();
	
	buildPointDrawBuf(msh.numVertices(), 
			(const float *)msh.triangleVertexP(), 
			(const float *)msh.triangleVertexN(),
			(const float *)msh.triangleVertexN() );
	buildBounding8Dop(bbox);
}
*/
void ExampVox::voxelize3(sdb::VectorArray<cvx::Triangle> * tri,
							const BoundingBox & bbox)
{
	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 32;
	KdEngine engine;
	KdNTree<cvx::Triangle, KdNode4 > gtr;
	engine.buildTree<cvx::Triangle, KdNode4, 4>(&gtr, tri, bbox, &bf);

	BoundingBox tb = gtr.getBBox();
	const float gz = tb.getLongestDistance() * .89f;
	
typedef IntersectEngine<cvx::Triangle, KdNode4 > FIntersectTyp;
	FIntersectTyp fintersect(&gtr);
	
typedef ClosestToPointEngine<cvx::Triangle, KdNode4 > FClosestTyp;
    FClosestTyp fclosest(&gtr);
	
	int mxLevel = 3;
	if(tri->size() > 8192) {
		mxLevel++;
	}
	
	sdb::AdaptiveGridDivideProfle subdprof;
	subdprof.setLevels(0, mxLevel);
	
	sdb::LodSampleCache spg;
	spg.fillBox(tb, gz);
	spg.subdivideToLevel<FIntersectTyp>(fintersect, subdprof);
	spg.insertNodeAtLevel<FClosestTyp, 3 >(mxLevel, fclosest);
	spg.buildSampleCache(mxLevel, mxLevel);
	
	const int & ns = spg.numSamplesAtLevel(mxLevel);
	std::cout<<"\n ExampVox::voxelize3 has n samples "<<ns;
	std::cout.flush();
	
	const sdb::SampleCache * sps = spg.samplesAtLevel(mxLevel);
	buildPointDrawBuf(ns, sps->points(), sps->normals(), sps->colors(),
					sdb::SampleCache::DataStride>>2);
/// obsolete
	//buildBounding8Dop(bbox);
	buildPointHull(bbox);
}

void ExampVox::buildPointHull(const BoundingBox & bbox)
{
    std::cout<<"\n buildPointHull "<<bbox;
	const int & np = pntBufLength();
	const Vector3F * pr = pntPositionR();
	const Vector3F * nr = pntNormalR();
	const float sz0 = bbox.getLongestDistance() * .73f;
	
	KHullGen<PosNml> khl;
	khl.fillBox(bbox, sz0);
	
	PosNml smp;
	for(int i=0;i<np;++i) {
	    
		smp._pos = pr[i];
		smp._nml = nr[i];
		
		khl.insertValueAtLevel(3, smp._pos, smp);
	}
	khl.finishInsert();
	
	int k = 5;
	if(khl.numCellsAtLevel(3) < 100) {
	     k = 2;
	}
	ATriangleMesh msh;
	khl.build(&msh, 3, k);
	
	const int nv = msh.numPoints();
	setDopDrawBufLen(nv);
	float * posf = dopRefPositionR();
	float * posf1 = dopPositionR();
	float * nmlf = dopNormalR();
	
	for(int i=0;i<nv;++i) {
		memcpy(&posf[i*3], &msh.points()[i], 12);
		memcpy(&posf1[i*3], &msh.points()[i], 12);
		memcpy(&nmlf[i*3], &msh.vertexNormals()[i], 12);
	}
}

void ExampVox::buildBounding8Dop(const BoundingBox & bbox)
{
	AOrientedBox ob;
	//Matrix33F zup;
	//zup.rotateX(-1.57f);
	//ob.setOrientation(zup);
	ob.caluclateOrientation(&bbox);
	ob.calculateCenterExtents(pntPositionBuf(), pntBufLength(), &bbox );
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
/// limit box (-1 0 -1 1 1 1)
	if(bx->distance(0) < 1e-1f) {
		bx->setMin(-1.f, 0);
		bx->setMax( 1.f, 0);
	}
	if(bx->distance(1) < 1e-1f) {
		bx->setMin( 0.f, 1);
		bx->setMax( 1.f, 1);
	}
	if(bx->distance(2) < 1e-1f) {
		bx->setMin(-1.f, 2);
		bx->setMax( 1.f, 2);
	}
	setGeomBox2(*bx);
}

void ExampVox::setGeomBox2(const BoundingBox & bx)
{
	m_geomBox = bx;
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

void ExampVox::setActive(bool x)
{ m_isActive = x; }

void ExampVox::setVisible(bool x)
{ m_isVisible = x; }

const bool & ExampVox::isActive() const
{ return m_isActive; }
	
const bool & ExampVox::isVisible() const
{ return m_isVisible; }

void ExampVox::setDiffuseMaterialCol(const float * x)
{ memcpy(m_diffuseMaterialColV, x, 12); }

void ExampVox::setDiffuseMaterialCol3(const float & xr,
					const float & xg,
					const float & xb)
{
	m_diffuseMaterialColV[0] = xr;
	m_diffuseMaterialColV[1] = xg;
	m_diffuseMaterialColV[2] = xb;
}

void ExampVox::updateDopCol()
{ setUniformDopColor(m_diffuseMaterialColV); }

void ExampVox::setDetailDrawType(const short & x)
{ m_drawDetailType  = x; }

const short& ExampVox::detailDrawType() const
{ return m_drawDetailType; }

void ExampVox::drawDetail() const
{
	if(m_drawDetailType == 1) {
		drawSolidGrid();
	} else {
		drawPoints();
	}
}

void ExampVox::drawFlatBound() const
{
	drawFlatSolidDop();
}

}
