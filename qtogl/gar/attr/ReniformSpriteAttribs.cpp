/*
 *  ReniformSpriteAttribs.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "ReniformSpriteAttribs.h"
#include <geom/ReniformMesh.h>
#include <gar_common.h>

using namespace aphid;

int ReniformSpriteAttribs::sNumInstances = 0;

ReniformSpriteAttribs::ReniformSpriteAttribs() : PieceAttrib(gar::gtReniformSprite)
{
	m_instId = sNumInstances;
	sNumInstances++;
	
	m_billboard = new ReniformMesh;
	
	addFloatAttrib(gar::nWidth, 4.f, 2.f, 100.f);
	addFloatAttrib(gar::nKidneyAngle, 4.f, 1.f, 6.f);
	addFloatAttrib(gar::nStalkHeight, 4.f, 2.f, 100.f);
	addIntAttrib(gar::nStalkWidth, 5, 2, 30);
	addFloatAttrib(gar::nMidribHeight, 1.f, .5f, 50.f);
	addIntAttrib(gar::nVeinSegments, 1, 1, 8);
	addFloatAttrib(gar::nAscendAngle, .5f, 0.f, 2.f);
	addSplineAttrib(gar::nLeftSide);
	addSplineAttrib(gar::nRightSide);
	addSplineAttrib(gar::nVein);
	addSplineAttrib(gar::nVeinVariation);
	
	gar::SplineAttrib* avs = (gar::SplineAttrib*)findAttrib(gar::nVein);
	avs->setSplineValue(.5f, .5f);
	avs->setSplineCv0(.4f, .5f);
	avs->setSplineCv1(.6f, .5f);
	update();
}

bool ReniformSpriteAttribs::hasGeom() const
{ return true; }
	
int ReniformSpriteAttribs::numGeomVariations() const
{ return 1; }

ATriangleMesh* ReniformSpriteAttribs::selectGeom(gar::SelectProfile* prof) const
{
	prof->_exclR = m_exclR;
	prof->_height = m_billboard->height();
	return m_billboard; 
}

bool ReniformSpriteAttribs::update()
{
	SplineMap1D* ls = m_billboard->leftSideSpline();
	SplineMap1D* rs = m_billboard->rightSideSpline();
	SplineMap1D* vs = m_billboard->veinSpline();
	SplineMap1D* vvs = m_billboard->veinVarySpline();
	
	gar::SplineAttrib* als = (gar::SplineAttrib*)findAttrib(gar::nLeftSide);
	gar::SplineAttrib* ars = (gar::SplineAttrib*)findAttrib(gar::nRightSide);
	gar::SplineAttrib* avs = (gar::SplineAttrib*)findAttrib(gar::nVein);
	gar::SplineAttrib* avvs = (gar::SplineAttrib*)findAttrib(gar::nVeinVariation);
	
	updateSplineValues(ls, als);
	updateSplineValues(rs, ars);
	updateSplineValues(vs, avs);
	updateSplineValues(vvs, avvs);
	
	ReniformMeshProfile prof;
	float w; 
	findAttrib(gar::nWidth)->getValue(w);
	findAttrib(gar::nStalkHeight)->getValue(prof._stalkHeight);
	findAttrib(gar::nKidneyAngle)->getValue(prof._kidneyAngle);
	int stalkW;
	findAttrib(gar::nStalkWidth)->getValue(stalkW);
	prof._stalkWidth = w * (float)stalkW * .01f;
	findAttrib(gar::nMidribHeight)->getValue(prof._midribHeight);
	findAttrib(gar::nVeinSegments)->getValue(prof._veinSegments);
	findAttrib(gar::nAscendAngle)->getValue(prof._ascendAngle);
	
	m_exclR = w * .47f;
	prof._radius = (w - prof._stalkWidth) * .5f; 
	prof._stalkSegments = 4 + .3f * prof._stalkHeight / prof._stalkWidth;
	if(prof._stalkSegments > 6)
		prof._stalkSegments = 6;
	
	m_billboard->createReniform(prof);
	return true;
}

int ReniformSpriteAttribs::attribInstanceId() const
{ return m_instId; }

bool ReniformSpriteAttribs::isGeomLeaf() const
{ return true; }

void ReniformSpriteAttribs::estimateExclusionRadius(float& minRadius)
{
	if(minRadius > m_exclR)
		minRadius = m_exclR;
}

bool ReniformSpriteAttribs::isGeomProfiled() const
{ return true; }
