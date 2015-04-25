/*
 *  CurveBuilder.cpp
 *  aphid
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "CurveBuilder.h"
std::vector<Vector3F> CurveBuilder::BuilderVertices;
CurveBuilder::CurveBuilder() {}

CurveBuilder::~CurveBuilder()
{ BuilderVertices.clear(); }

void CurveBuilder::addVertex(const Vector3F & vert)
{ BuilderVertices.push_back(vert); }

void CurveBuilder::finishBuild(BaseCurve * c)
{
	const unsigned n = (unsigned)BuilderVertices.size();
	c->createVertices(n);

	for(unsigned i = 0; i < n; i++) c->m_cvs[i] = BuilderVertices[i];
	
	c->computeKnots();
	
	BuilderVertices.clear();
}