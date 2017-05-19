/*
 *  GardenExamp.cpp
 *  rhizoid
 *
 *  Created by jian zhang on 5/12/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GardenExamp.h"
#include "CompoundExamp.h"
#include "SelectExmpCondition.h"

namespace aphid {

GardenExamp::GardenExamp()
{}

GardenExamp::~GardenExamp()
{}

void GardenExamp::addAExample(CompoundExamp * v)
{ m_examples.push_back(v); }

int GardenExamp::numExamples() const
{ return m_examples.size(); }

CompoundExamp * GardenExamp::getCompoundExample(const int & i)
{ return m_examples[i]; }

const ExampVox * GardenExamp::getExample(const int & i) const
{ return m_examples[i]; }

ExampVox * GardenExamp::getExample(const int & i)
{ return m_examples[i]; }


bool GardenExamp::isVariable() const
{ return true; }

int GardenExamp::selectExample(SelectExmpCondition & cond) const
{
	
	if(getPattern() == pnAngleAlign) {
		return fitToSurface(cond);
	}

	return rand() % numExamples(); 
}

int GardenExamp::fitToSurface(SelectExmpCondition & cond) const
{
	Vector3F surfNml = cond.surfaceNormal();
	surfNml.normalize();
	
	Matrix44F tm = cond.transform();
	
	Vector3F spaceUp = tm.getUp();
	spaceUp.normalize();
	Vector3F side = tm.getSide();
	float sz = side.length();
	side.normalize();
	
	int res = Variform::selectByAngle(surfNml, spaceUp, side);
	
	Vector3F up = surfNml;
	
	Vector3F front = side.cross(up);
	front.normalize();
	side = up.cross(front);
	tm.setOrientations(side, up, front);
	tm.scaleBy(sz);
	cond.setTransform(tm);
	return res;
}

}