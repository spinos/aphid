/*
 *  SplineCylinderAttribs.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "SplineCylinderAttribs.h"
#include <geom/SplineCylinder.h>
#include <gar_common.h>

using namespace aphid;

int SplineCylinderAttribs::sNumInstances = 0;

SplineCylinderAttribs::SplineCylinderAttribs() : PieceAttrib(gar::gtSplineCylinder)
{
	m_instId = sNumInstances;
	sNumInstances++;
	
	m_cylinder = new SplineCylinder;
	
	addFloatAttrib(gar::nRadius, 1.f, .5f, 25.f);
	addFloatAttrib(gar::nHeight, 20.f, 4.f, 200.f);
	addIntAttrib(gar::nAddSegment, 0, -20, 20);
	addSplineAttrib(gar::nRadiusVariation);
	addSplineAttrib(gar::nHeightVariation);
	update();
}

bool SplineCylinderAttribs::hasGeom() const
{ return true; }
	
int SplineCylinderAttribs::numGeomVariations() const
{ return 1; }

ATriangleMesh* SplineCylinderAttribs::selectGeom(gar::SelectProfile* prof) const
{
	prof->_exclR = m_exclR;
	prof->_height = m_cylinder->height();
	return m_cylinder; 
}

bool SplineCylinderAttribs::update()
{
	SplineMap1D* ls = m_cylinder->radiusSpline();
	SplineMap1D* rs = m_cylinder->heightSpline();
	
	gar::SplineAttrib* als = (gar::SplineAttrib*)findAttrib(gar::nRadiusVariation);
	gar::SplineAttrib* ars = (gar::SplineAttrib*)findAttrib(gar::nHeightVariation);
	
	updateSplineValues(ls, als);
	updateSplineValues(rs, ars);
	
	float r, h;
	findAttrib(gar::nRadius)->getValue(r);
	findAttrib(gar::nHeight)->getValue(h);
	int ag;
	findAttrib(gar::nAddSegment)->getValue(ag);
	
	int nv = 4 + h * .23f / r + ag;
	if(nv < 4)
		nv = 4;
	m_cylinder->createCylinder(5, nv, r, h);
	
	m_exclR = r * 1.9f;
	return true;
}

int SplineCylinderAttribs::attribInstanceId() const
{ return m_instId; }

bool SplineCylinderAttribs::isGeomStem() const
{ return true; }

void SplineCylinderAttribs::estimateExclusionRadius(float& minRadius)
{
	if(minRadius > m_exclR)
		minRadius = m_exclR;
}
