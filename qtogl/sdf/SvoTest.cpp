/*
 *  SvoTest.cpp
 *  
 *
 *  Created by jian zhang on 2/14/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "SvoTest.h"
#include <ttg/SparseVoxelOctree.h>
#include <ttg/LegendreSVORule.h>
#include <geom/SuperShape.h>
#include <GeoDrawer.h>
#include <kd/ClosestToPointEngine.h>
#include <smp/Triangle.h>
#include "PosSample.h"

SvoTest::SvoTest()
{
	m_shape = new SuperShapeGlyph; 
	m_traverser = new OctreeTraverseTyp;
	m_pnts = new PntArrTyp;
	m_hilbertRule = new SvoRuleTyp;
	m_drawCtx = new DrawCtxTyp(m_hilbertRule);
	m_rayCtx = new RayCtxTyp;
	
}

SuperFormulaParam& SvoTest::shapeParam()
{ return m_shape->param(); }

void SvoTest::drawShape(GeoDrawer * dr)
{
	dr->triangleMesh(m_shape);
}

void SvoTest::updateShape()
{ m_shape->computePositions(); }

const sdb::FHilbertRule& SvoTest::hilbertSFC() const
{ return m_hilbertRule->sfc(); }

ttg::LegendreSVORule<sdb::FHilbertRule>& SvoTest::svoRule()
{ return *m_hilbertRule; }

void SvoTest::sampleShape(BoundingBox& shapeBox)
{
	sdb::VectorArray<cvx::Triangle> tris;
	
	shapeBox.reset();
	KdEngine eng;
	eng.appendSource<cvx::Triangle, ATriangleMesh >(&tris, shapeBox,
									m_shape, 0);
	shapeBox.round();
	
	const float ssz = shapeBox.getLongestDistance() * .00191f;
	smp::Triangle sampler;
	sampler.setSampleSize(ssz);
	
	SampleInterp interp;
	
	m_pnts->clear();
	
	PosSample asmp;
/// same radius
	asmp._r = ssz;
	const int nt = tris.size();
	for(int i=0;i<nt;++i) {
		
		const cvx::Triangle* ti = tris.get(i);
		
		sampleTriangle(asmp, sampler, interp, ti);
	}
	
	std::cout<<"\n n triangle samples "<<m_pnts->size();
	
	const Vector3F midP = shapeBox.center();
	const float spanL = shapeBox.getLongestDistance();
	const float spanH = spanL * .5f;
	const Vector3F lowP(midP.x - spanH, 
						midP.y - spanH, 
						midP.z - spanH );
						
	sdb::FHilbertRule& hilbert = m_hilbertRule->sfc();
	hilbert.setOriginSpan(lowP.x, 
						lowP.y, 
						lowP.z,
						spanL);
	m_pnts->createSFC<sdb::FHilbertRule>(hilbert);
	
	std::cout<<"\n\n test svo ";
	
	m_hilbertRule->setMaxLevel(5);
	ttg::SVOBuilder<ttg::SVOBNode> bldsvo;
	bldsvo.build<PosSample, SvoRuleTyp >(pnts(), svoRule() );

	bldsvo.save<ttg::SVOTNode, SvoRuleTyp >(*m_traverser);
	std::cout<<"\n done svo \n\n";

}

void SvoTest::sampleTriangle(PosSample& asmp, smp::Triangle& sampler, 
						SampleInterp& interp, const cvx::Triangle* g)
{
	const int ns = sampler.getNumSamples(g->calculateArea() );
	int n = ns;
	for(int i=0;i<500;++i) {
		
		if(n < 1)
			return;
			
		if(!sampler.sampleTriangle<PosSample, SampleInterp >(asmp, interp, g) )
			continue;
			
		m_pnts->push_back(asmp);
		n--;
	}
}

sdb::SpaceFillingVector<PosSample>& SvoTest::pnts()
{ return *m_pnts; }

const sdb::SpaceFillingVector<PosSample>& SvoTest::pnts() const
{ return *m_pnts; }

void SvoTest::drawShapeSamples(GeoDrawer * dr, const int* drawRange) const
{
	const PntArrTyp& rpnts = pnts();
	const int ns = rpnts.size();
	if(ns < 1)
		return;
		
	glBegin(GL_POINTS);
	
#if 0
	glColor3f(.9f,.6f,0.f);

	for(int i=drawRange[0];i<drawRange[1];++i) {
		const PosSample& vi = rpnts[i];
		glVertex3fv((const float* )&(vi._pos));
	}
	
#else
	glColor3f(0.f,.4f,.3f);
	
	for(int i=0;i<ns;++i) {
		const PosSample& vi = rpnts[i];
		
		if(vi._key >= drawRange[0]) 
			glColor3f(.9f,.6f,0.f);
		if(vi._key >= drawRange[1])
			glColor3f(0.f,.4f,.3f);
	
		glVertex3fv((const float* )&(vi._pos));
	}
#endif

	glEnd();
}

void SvoTest::drawSVO(aphid::GeoDrawer * dr)
{
	if(m_traverser->numNodes() < 1)
		return;
	
	glColor3f(0.f,.23f,.3f);	
	BoundingBox bx;
	m_drawCtx->begin(m_traverser->nodes(), m_traverser->coord() );
	while(!m_drawCtx->end() ) {
	
		const float* coord = m_drawCtx->currentCoord();
		bx.setCenterHalfSpan(coord);
		dr->boundingBox(bx);
		
		m_drawCtx->next();
	}
}
