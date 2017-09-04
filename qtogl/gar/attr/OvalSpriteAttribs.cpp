/*
 *  OvalSpriteAttribs.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "OvalSpriteAttribs.h"
#include <geom/EllipseMesh.h>
#include <gar_common.h>

using namespace aphid;

int OvalSpriteAttribs::sNumInstances = 0;

OvalSpriteAttribs::OvalSpriteAttribs() : PieceAttrib(gar::gtOvalSprite)
{
	m_instId = sNumInstances;
	sNumInstances++;
	
	m_billboard = new EllipseMesh;
	
	addFloatAttrib(gar::nWidth, 4.f, 2.f, 80.f);
	addFloatAttrib(gar::nHeight, 6.f, 3.f, 120.f);
	addIntAttrib(gar::nNProfiles, 1, 1, 5);
	addIntAttrib(gar::nAddSegment, 0, -20, 20);
	addIntAttrib(gar::nMidribWidth, 10, 5, 50);
	addSplineAttrib(gar::nLeftSide);
	addSplineAttrib(gar::nRightSide);
	addSplineAttrib(gar::nHeightVariation);
	addSplineAttrib(gar::nVein);
	addSplineAttrib(gar::nVeinVariation);
	gar::SplineAttrib* avs = (gar::SplineAttrib*)findAttrib(gar::nVein);
	avs->setSplineValue(.5f, .5f);
	avs->setSplineCv0(.4f, .5f);
	avs->setSplineCv1(.6f, .5f);
	update();
}

bool OvalSpriteAttribs::hasGeom() const
{ return true; }
	
int OvalSpriteAttribs::numGeomVariations() const
{ return 1; }

ATriangleMesh* OvalSpriteAttribs::selectGeom(gar::SelectProfile* prof) const
{
	prof->_exclR = m_exclR;
	prof->_height = m_billboard->height();
	return m_billboard; 
}

bool OvalSpriteAttribs::update()
{
	SplineMap1D* ls = m_billboard->leftSpline();
	SplineMap1D* rs = m_billboard->rightSpline();
	SplineMap1D* vs = m_billboard->veinSpline();
	SplineMap1D* vvs = m_billboard->veinVarySpline();
	SplineMap1D* hs = m_billboard->heightSpline();
	
	gar::SplineAttrib* als = (gar::SplineAttrib*)findAttrib(gar::nLeftSide);
	gar::SplineAttrib* ars = (gar::SplineAttrib*)findAttrib(gar::nRightSide);
	gar::SplineAttrib* avs = (gar::SplineAttrib*)findAttrib(gar::nVein);
	gar::SplineAttrib* avvs = (gar::SplineAttrib*)findAttrib(gar::nVeinVariation);
	gar::SplineAttrib* ahs = (gar::SplineAttrib*)findAttrib(gar::nHeightVariation);
	
	updateSplineValues(ls, als);
	updateSplineValues(rs, ars);
	updateSplineValues(vs, avs);
	updateSplineValues(vvs, avvs);
	updateSplineValues(hs, ahs);
	
	float w, h;
	findAttrib(gar::nWidth)->getValue(w);
	findAttrib(gar::nHeight)->getValue(h);
	int nu = 1;
	findAttrib(gar::nNProfiles)->getValue(nu);
		
	m_exclR = w * .47f;
	int ag;
	findAttrib(gar::nAddSegment)->getValue(ag);
	int nv = 4 + (h * nu) / w + ag;
	if(nv < 4)
		nv = 4;
	
	m_billboard->createEllipse(w, h, nv, nu);
	return true;
}

int OvalSpriteAttribs::attribInstanceId() const
{ return m_instId; }

bool OvalSpriteAttribs::isGeomLeaf() const
{ return true; }

void OvalSpriteAttribs::estimateExclusionRadius(float& minRadius)
{
	if(minRadius > m_exclR)
		minRadius = m_exclR;
}

bool OvalSpriteAttribs::isGeomProfiled() const
{ return true; }
