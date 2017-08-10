/*
 *  BendTwistRollAttribs.cpp
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "BendTwistRollAttribs.h"
#include <geom/ATriangleMesh.h>

using namespace aphid;

BendTwistRollAttribs::BendTwistRollAttribs() :
m_inGeom(NULL)
{
	addFloatAttrib(gar::nBend, 1.f, 0.f, 1.f);
	addFloatAttrib(gar::nTwist, 0.5f, 0.f, 1.f);
	addFloatAttrib(gar::nRoll, 0.2f, 0.f, 1.f);
}

void BendTwistRollAttribs::setInputGeom(ATriangleMesh* x)
{ m_inGeom = x; }

bool BendTwistRollAttribs::hasGeom() const
{ return m_inGeom != NULL; }

int BendTwistRollAttribs::numGeomVariations() const
{ return 64; }

ATriangleMesh* BendTwistRollAttribs::selectGeom(int x, float& exclR) const
{
	return m_outGeom[x]; 
}

bool BendTwistRollAttribs::update()
{
	return false;
}

int BendTwistRollAttribs::attribInstanceId() const
{ return 0; }
