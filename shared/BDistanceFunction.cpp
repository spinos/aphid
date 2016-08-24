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
	std::vector<Domain *>::iterator it = m_domains.begin();
	for(;it!=m_domains.end();++it) {
		delete *it;
	}
	m_domains.clear();
}

void BDistanceFunction::addSphere(const Vector3F & p, const float & r)
{
	cvx::Sphere * s = new cvx::Sphere;
	s->set(p, r);
	m_domains.push_back(new SphereDomain(s) );
}

void BDistanceFunction::addBox(const Vector3F & lo, const Vector3F & hi)
{
	cvx::Box * s = new cvx::Box;
	s->set(lo, hi);
	m_domains.push_back(new BoxDomain(s) );
}

void BDistanceFunction::setDomainDistanceRange(const float & x)
{
    std::vector<Domain *>::iterator it = m_domains.begin();
	for(;it!=m_domains.end();++it) {
		
		(*it)->setDistanceRange(x);
	}
}

float BDistanceFunction::calculateDistance(const Vector3F & p)
{
#define DISTANCE_POSITIVEINF 1e8f;	
	float d, mnd = DISTANCE_POSITIVEINF;
	std::vector<Domain *>::iterator it = m_domains.begin();
	for(;it!=m_domains.end();++it) {
		
		d = (*it)->distanceTo(p);
		if(mnd > d)
			mnd = d;
			
	}
	return mnd;
}

}