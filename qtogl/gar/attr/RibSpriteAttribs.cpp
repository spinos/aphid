/*
 *  RibSpriteAttribs.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "RibSpriteAttribs.h"
#include <geom/SplineBillboard.h>

using namespace aphid;

int RibSpriteAttribs::sNumInstances = 0;

RibSpriteAttribs::RibSpriteAttribs()
{
	m_instId = sNumInstances;
	sNumInstances++;
	
	m_billboard = new SplineBillboard;
	m_billboard->setBillboardSize(4.f, 6.f, 2);
	
	addFloatAttrib(gar::nWidth, 4.f, 2.f, 80.f);
	addFloatAttrib(gar::nHeight, 6.f, 3.f, 120.f);
	addSplineAttrib(gar::nCenterLine);
	addSplineAttrib(gar::nLeftSide);
	addSplineAttrib(gar::nRightSide);
	
	gar::SplineAttrib* acs = (gar::SplineAttrib*)findAttrib(gar::nCenterLine);
	acs->setSplineValue(.5f, .5f);
	acs->setSplineCv0(.4f, .5f);
	acs->setSplineCv1(.6f, .5f);
	
	update();
}

bool RibSpriteAttribs::hasGeom() const
{ return true; }
	
int RibSpriteAttribs::numGeomVariations() const
{ return 1; }

ATriangleMesh* RibSpriteAttribs::selectGeom(int x, float& exclR) const
{
	const gar::Attrib* wa = findAttrib(gar::nWidth);
	wa->getValue(exclR);
	exclR *= .4f;
	return m_billboard; 
}

bool RibSpriteAttribs::update()
{
	SplineMap1D* cs = m_billboard->centerSpline();
	SplineMap1D* ls = m_billboard->leftSpline();
	SplineMap1D* rs = m_billboard->rightSpline();
	
	gar::SplineAttrib* acs = (gar::SplineAttrib*)findAttrib(gar::nCenterLine);
	gar::SplineAttrib* als = (gar::SplineAttrib*)findAttrib(gar::nLeftSide);
	gar::SplineAttrib* ars = (gar::SplineAttrib*)findAttrib(gar::nRightSide);
	
	updateSplineValues(cs, acs);
	updateSplineValues(ls, als);
	updateSplineValues(rs, ars);
	
	float w, h;
	findAttrib(gar::nWidth)->getValue(w);
	findAttrib(gar::nHeight)->getValue(h);
	
	m_billboard->setBillboardSize(w, h, 2);
	return true;
}

int RibSpriteAttribs::attribInstanceId() const
{ return m_instId; }

float RibSpriteAttribs::texcoordBlockAspectRatio() const
{ return m_billboard->widthHeightRatio(); }
