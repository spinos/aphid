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
#include <gar_common.h>

using namespace aphid;

int SplineSpriteAttribs::sNumInstances = 0;

SplineSpriteAttribs::SplineSpriteAttribs() : PieceAttrib(gar::gtSplineSprite)
{
	m_instId = sNumInstances;
	sNumInstances++;
	
	m_billboard = new SplineBillboard;
	m_billboard->setBillboardSize(4.f, 6.f, 1);
	
	addFloatAttrib(gar::nWidth, 4.f, 2.f, 80.f);
	addFloatAttrib(gar::nHeight, 6.f, 3.f, 120.f);
	addSplineAttrib(gar::nLeftSide);
	addSplineAttrib(gar::nRightSide);
	update();
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
	
	updateSplineValues(ls, als);
	updateSplineValues(rs, ars);
	
	float w, h;
	findAttrib(gar::nWidth)->getValue(w);
	findAttrib(gar::nHeight)->getValue(h);
	
	m_billboard->setBillboardSize(w, h, 1);
	return true;
}

int SplineSpriteAttribs::attribInstanceId() const
{ return m_instId; }

float SplineSpriteAttribs::texcoordBlockAspectRatio() const
{ return m_billboard->widthHeightRatio(); }
