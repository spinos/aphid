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

using namespace aphid;

int SplineCylinderAttribs::sNumInstances = 0;

SplineCylinderAttribs::SplineCylinderAttribs()
{
	m_instId = sNumInstances;
	sNumInstances++;
	
	m_cylinder = new SplineCylinder;
	
	addFloatAttrib(gar::nRadius, 2.f, 1.f, 80.f);
	addFloatAttrib(gar::nHeight, 10.f, 4.f, 120.f);
	addSplineAttrib(gar::nRadiusVariation);
	addSplineAttrib(gar::nHeightVariation);
}

bool SplineCylinderAttribs::hasGeom() const
{ return true; }
	
int SplineCylinderAttribs::numGeomVariations() const
{ return 1; }

ATriangleMesh* SplineCylinderAttribs::selectGeom(int x, float& exclR) const
{
	const gar::Attrib* wa = findAttrib(gar::nRadius);
	wa->getValue(exclR);
	return m_cylinder; 
}

bool SplineCylinderAttribs::update()
{
	SplineMap1D* ls = m_cylinder->radiusSpline();
	SplineMap1D* rs = m_cylinder->heightSpline();
	
	gar::SplineAttrib* als = (gar::SplineAttrib*)findAttrib(gar::nRadiusVariation);
	gar::SplineAttrib* ars = (gar::SplineAttrib*)findAttrib(gar::nHeightVariation);
	
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
	
	float r, h;
	findAttrib(gar::nRadius)->getValue(r);
	findAttrib(gar::nHeight)->getValue(h);
	
	int nv = 4 + h * .19f / r;
	m_cylinder->createCylinder(6, nv, r, h);
	return true;
}

int SplineCylinderAttribs::attribInstanceId() const
{ return m_instId; }

float SplineCylinderAttribs::texcoordBlockAspectRatio() const
{ return m_cylinder->circumferenceHeightRatio(); }
