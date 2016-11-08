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

BDistanceFunction::BDistanceFunction() :
m_shellThickness(0.f),
m_splatRadius(0.f)
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

void BDistanceFunction::setShellThickness(const float & x)
{ m_shellThickness = x; }

void BDistanceFunction::setSplatRadius(const float & x)
{ m_splatRadius = x; }

const float & BDistanceFunction::shellThickness() const
{ return m_shellThickness; }

const float & BDistanceFunction::splatRadius() const
{ return m_splatRadius; }

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
		
		d = (*it)->distanceTo(p) - shellThickness();
		if(mnd > d)
			mnd = d;
			
	}
	return mnd;
}

float BDistanceFunction::calculateIntersection(const Vector3F & a,
								const Vector3F & b,
								const float & ra,
								const float & rb)
{
	Ray r(a, b);
	Beam bm(r, ra, rb);
	float md = 1e8f, d;
	std::vector<Domain *>::iterator it = m_domains.begin();
	for(;it!=m_domains.end();++it) {
		
		d = (*it)->beamIntersect(bm, splatRadius() );
		
		if(md > d)
			md = d;
			
	}
	
	if(md > .99999e8f)
		return -1;
		
	return md;
}

bool BDistanceFunction::narrowphase(const cvx::Hexahedron & a) const
	{
		std::vector<Domain *>::const_iterator it = m_domains.begin();
		for(;it!=m_domains.end();++it) {
			
			Domain * dm = *it;
			bool stat = dm->narrowphaseHexahedron (a);
			
			if(stat)
				return true;
			
		}
		return false;
	}


}