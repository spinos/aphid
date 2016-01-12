/*
 *  ActiveGroup.cpp
 *  btree
 *
 *  Created by jian zhang on 1/12/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ActiveGroup.h"

namespace sdb {

ActiveGroup::ActiveGroup() 
{ 
	vertices = new Array<int, VertexP>; 
	reset();
	m_drop = new DropoffCosineCurve;
	m_dropoffType = Dropoff::kCosineCurve;
	m_volumeType = VSphere;
}

ActiveGroup::~ActiveGroup() 
{
    delete m_drop;
	vertices->clear();
	delete vertices;
}

void ActiveGroup::reset() 
{
	depthMin = 10e8;
	vertices->clear();
	meanPosition.setZero();
	meanNormal.setZero();
}

int ActiveGroup::numSelected() 
{ return vertices->size(); }

void ActiveGroup::updateMinDepth(float d)
{ if(depthMin > d) depthMin = d; }

float ActiveGroup::minDepth() const
{ return depthMin; }

void ActiveGroup::finish() 
{
	const int nsel = vertices->size();
	if(nsel < 1) return;

	average(vertices);
	
	if(nsel > 1) {
		meanPosition *= 1.f / (float)nsel;
		
		meanPosition = incidentRay.closetPointOnRay(meanPosition);
		
		meanNormal.normalize();
	}
	//std::cout<<"\n n sel "<<nsel
	//<<"\n mean p "<<meanPosition<<" depth "<<depthMin;
	
	calculateWeight(vertices);
}

void ActiveGroup::average(Array<int, VertexP> * d)
{
	const int num = d->size();
	if(num < 1) return;
	d->begin();
	while(!d->end()) {
		//if(n->dot(incidentRay.m_dir) < 0.f) {
			meanPosition += *d->value()->index->t1;
			meanNormal += *d->value()->index->t2;
		//}
		d->next();
	}
}

void ActiveGroup::calculateWeight(Array<int, VertexP> * d)
{
	float wei;
    d->begin();
	while(!d->end()) {
		Vector3F * p = d->value()->index->t1;
		if(m_volumeType==VCylinder) {
			const Vector3F por = incidentRay.closetPointOnRay(*p);
			wei = m_drop->f(p->distanceTo(por), threshold);
		}
		else 
			wei = m_drop->f(p->distanceTo(meanPosition), threshold);
			
		*d->value()->index->t4 = wei;
		d->next();
	}
}

void ActiveGroup::setDropoffFunction(Dropoff::DistanceFunction x)
{
    if(x == m_dropoffType) return;
    delete m_drop;
    m_dropoffType = x;
    switch(x) {
        case Dropoff::kQuadratic :
            m_drop = new DropoffQuadratic;
            break;
        case Dropoff::kCubic :
            m_drop = new DropoffCubic;
            break;
		case Dropoff::kCosineCurve :
            m_drop = new DropoffCosineCurve;
            break;
        case Dropoff::kExponential :
            m_drop = new DropoffExponential;
            break;
        default:
            m_drop = new DropoffLinear;
            break;
    }
}

const Dropoff::DistanceFunction ActiveGroup::dropoffFunction() const
{ return m_dropoffType; }

void ActiveGroup::setSelectVolume(VolumeType t )
{ m_volumeType = t; }

}