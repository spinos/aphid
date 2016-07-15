/*
 *  BDistanceFunction.cpp
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "BDistanceFunction.h"

namespace aphid {

BDistanceFunction::BDistanceFunction()
{}

BDistanceFunction::~BDistanceFunction()
{ internalClear(); }

void BDistanceFunction::internalClear()
{
	std::vector<cvx::Shape *>::iterator it = m_shapes.begin();
	for(;it!=m_shapes.end();++it) {
		delete *it;
	}
	m_shapes.clear();
}

void BDistanceFunction::addSphere(const Vector3F & p, const float & r)
{
	cvx::Sphere * s = new cvx::Sphere;
	s->set(p, r);
	addFunction(s);
}

void BDistanceFunction::addBox(const Vector3F & lo, const Vector3F & hi)
{
	cvx::Box * s = new cvx::Box;
	s->set(lo, hi);
	addFunction(s);
}

void BDistanceFunction::addFunction(cvx::Shape * d)
{ m_shapes.push_back(d); }

float BDistanceFunction::calculateDistance(const Vector3F & p) const
{
#define DISTANCE_POSITIVEINF 1e8f;	
	float d, mnd = DISTANCE_POSITIVEINF;
	std::vector<cvx::Shape *>::const_iterator it = m_shapes.begin();
	for(;it!=m_shapes.end();++it) {
		
		d = (*it)->distanceTo(p);
		if(mnd > d)
			mnd = d;
	}
	return mnd;
}

}