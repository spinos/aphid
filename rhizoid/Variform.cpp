/*
 *  Variform.cpp
 *  
 *
 *  Created by jian zhang on 5/20/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "Variform.h"
#include <math/Matrix44F.h>
#include <stdlib.h>
#include <cmath>

namespace aphid {

int Variform::NumAngleGroups = 8;
int Variform::NumEventsPerGroup = 8;

Variform::Variform() :
m_pattern(pnRandom)
{}

Variform::~Variform()
{}

void Variform::setPattern(Pattern x)
{ m_pattern = x; }

void Variform::setShortPattern(short x)
{
	switch(x) {
		case 1:
			setPattern(pnAngleAlign);
		break;
		default:
			setPattern(pnRandom);
		;
	}
}

const Variform::Pattern & Variform::getPattern() const
{ return m_pattern; }

float Variform::deltaAnglePerGroup()
{ return .8f / ((float)NumAngleGroups - 1); }

int Variform::selectByAngle(const Vector3F & surfaceNml,
				const Vector3F & frameUp,
				Vector3F & modSide) const
{
	float ang = acos(surfaceNml.dot(frameUp) );
	if(ang > .8f) {
		ang = .8f;
	}
	
	int grp0 = .5f + ang / deltaAnglePerGroup();
	if(grp0 > NumAngleGroups - 1) {
		grp0 = NumAngleGroups - 1;
	}
	
	if(grp0 > 0) {
		Vector3F frmSide = frameUp.cross(surfaceNml);
		frmSide.normalize();
		
		float wei = sqrt(ang / .8f);
		rotateSide(modSide, frmSide, wei);
		modSide = frmSide;
	}
	
	return grp0 * NumEventsPerGroup + rand() % NumEventsPerGroup;
}

void Variform::rotateSide(Vector3F & modSide,
				const Vector3F & frameSide,
				const float & alpha) const
{
	float ang = acos(modSide.dot(frameSide) ) * alpha;
	if(ang < .09f) {
		return;
	}
	
	Vector3F vUp = modSide.cross(frameSide);
	vUp.normalize();
	
	Vector3F vFront = modSide.cross(vUp);
	vFront.normalize();
	
	Matrix44F frm;
	frm.setOrientations(modSide, vUp, vFront);
	
	Matrix33F srot = frm.rotation();
	Quaternion q(ang, vUp);
	Matrix33F eft(q);
	
	srot *= eft;
	
	frm.setRotation(srot);
	modSide = frm.getSide();
}

}
