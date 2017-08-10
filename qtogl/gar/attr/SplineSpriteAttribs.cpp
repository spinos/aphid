/*
 *  SplineSpriteAttribs.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "SplineSpriteAttribs.h"
#include <geom/SplineBillboard.h>

using namespace aphid;

int SplineSpriteAttribs::sNumInstances = 0;

SplineSpriteAttribs::SplineSpriteAttribs()
{
	m_instId = sNumInstances;
	sNumInstances++;
	
	m_billboard = new SplineBillboard(4.f, 6.f);
	
	addFloatAttrib(gar::nWidth, 4.f, 2.f, 80.f);
	addFloatAttrib(gar::nHeight, 6.f, 3.f, 120.f);
	addSplineAttrib(gar::nLeftSide);
	addSplineAttrib(gar::nRightSide);
}

bool SplineSpriteAttribs::hasGeom() const
{ return true; }
	
int SplineSpriteAttribs::numGeomVariations() const
{ return 1; }

ATriangleMesh* SplineSpriteAttribs::selectGeom(int x, float& exclR) const
{
	const gar::Attrib* wa = findAttrib(gar::nWidth);
	wa->getValue(exclR);
	exclR *= .4f;
	return m_billboard; 
}

bool SplineSpriteAttribs::update()
{
	SplineMap1D* ls = m_billboard->leftSpline();
	SplineMap1D* rs = m_billboard->rightSpline();
	
	gar::SplineAttrib* als = (gar::SplineAttrib*)findAttrib(gar::nLeftSide);
	gar::SplineAttrib* ars = (gar::SplineAttrib*)findAttrib(gar::nRightSide);
	
	float tmp[2];
	als->getSplineValue(tmp);
	ls->setStart(tmp[0]);
	ls->setEnd(tmp[1]);
	
	als->getSplineCv0(tmp);
	ls->setLeftControl(tmp[0], tmp[1]);
	
	als->getSplineCv1(tmp);
	ls->setRightControl(tmp[0], tmp[1]);
	
	ars->getSplineValue(tmp);
	rs->setStart(tmp[0]);
	rs->setEnd(tmp[1]);
	
	ars->getSplineCv0(tmp);
	rs->setLeftControl(tmp[0], tmp[1]);
	
	ars->getSplineCv1(tmp);
	rs->setRightControl(tmp[0], tmp[1]);
	
	float w, h;
	findAttrib(gar::nWidth)->getValue(w);
	findAttrib(gar::nHeight)->getValue(h);
	
	m_billboard->setBillboardSize(w, h);
	return true;
}

int SplineSpriteAttribs::attribInstanceId() const
{ return m_instId; }
