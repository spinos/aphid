/*
 *  BladeSpriteAttribs.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "BladeSpriteAttribs.h"
#include <geom/SplineBlade.h>
#include <gar_common.h>

using namespace aphid;

int BladeSpriteAttribs::sNumInstances = 0;

BladeSpriteAttribs::BladeSpriteAttribs() : PieceAttrib(gar::gtSplineSprite)
{
	m_instId = sNumInstances;
	sNumInstances++;
	
	m_billboard = new SplineBlade;
	
	addFloatAttrib(gar::nWidth, 4.f, 2.f, 80.f);
	addFloatAttrib(gar::nHeight, 6.f, 3.f, 120.f);
	addIntAttrib(gar::nNProfiles, 4, 4, 10);
	addIntAttrib(gar::nAddSegment, 0, -20, 20);
	addIntAttrib(gar::nMidribWidth, 10, 5, 50);
	addSplineAttrib(gar::nLeftSide);
	addSplineAttrib(gar::nRightSide);
	addSplineAttrib(gar::nVein);
	gar::SplineAttrib* avs = (gar::SplineAttrib*)findAttrib(gar::nVein);
	avs->setSplineValue(.5f, .5f);
	avs->setSplineCv0(.4f, .5f);
	avs->setSplineCv1(.6f, .5f);
	update();
}

bool BladeSpriteAttribs::hasGeom() const
{ return true; }
	
int BladeSpriteAttribs::numGeomVariations() const
{ return 1; }

ATriangleMesh* BladeSpriteAttribs::selectGeom(gar::SelectProfile* prof) const
{
	prof->_exclR = m_exclR;
	prof->_height = m_billboard->height();
	return m_billboard; 
}

bool BladeSpriteAttribs::update()
{
	SplineMap1D* ls = m_billboard->leftSpline();
	SplineMap1D* rs = m_billboard->rightSpline();
	SplineMap1D* vs = m_billboard->veinSpline();
	
	gar::SplineAttrib* als = (gar::SplineAttrib*)findAttrib(gar::nLeftSide);
	gar::SplineAttrib* ars = (gar::SplineAttrib*)findAttrib(gar::nRightSide);
	gar::SplineAttrib* avs = (gar::SplineAttrib*)findAttrib(gar::nVein);
	
	updateSplineValues(ls, als);
	updateSplineValues(rs, ars);
	updateSplineValues(vs, avs);
	
	float w, h;
	findAttrib(gar::nWidth)->getValue(w);
	findAttrib(gar::nHeight)->getValue(h);
	int nu = 4;
	findAttrib(gar::nNProfiles)->getValue(nu);
	if(nu & 1)
		nu++;
		
	m_exclR = w * .47f;
	int ag;
	findAttrib(gar::nAddSegment)->getValue(ag);
	int nv = 4 + .43f * h / w + ag;
	if(nv < 4)
		nv = 4;
	float tip = .97f * h / (float)(nv+1);
	
	int ribw = 10;
	findAttrib(gar::nMidribWidth)->getValue(ribw);
	float fribw = (float)ribw * .01f * w;
	
	m_billboard->createBlade(w, h, fribw, tip,
								nv, nu);
	return true;
}

int BladeSpriteAttribs::attribInstanceId() const
{ return m_instId; }

bool BladeSpriteAttribs::isGeomLeaf() const
{ return true; }

void BladeSpriteAttribs::estimateExclusionRadius(float& minRadius)
{
	if(minRadius > m_exclR)
		minRadius = m_exclR;
}

bool BladeSpriteAttribs::isGeomProfiled() const
{ return true; }
