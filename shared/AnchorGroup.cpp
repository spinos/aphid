/*
 *  AnchorGroup.cpp
 *  masq
 *
 *  Created by jian zhang on 4/15/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "AnchorGroup.h"

AnchorGroup::AnchorGroup() {m_activeAnchor = 0; m_hitTolerance = 0.5f;}
AnchorGroup::~AnchorGroup() {}

void AnchorGroup::reset()
{
	m_activeAnchor = 0;
	for(std::vector<Anchor *>::iterator it = m_anchors.begin(); it != m_anchors.end(); ++it) {
		delete *it;
	}
	m_anchors.clear();
}

void AnchorGroup::addAnchor(Anchor * a)
{
	m_anchors.push_back(a);
}

bool AnchorGroup::pickupAnchor(const Ray & ray, Vector3F & hit)
{
	m_activeAnchor = 0;
	float minDist = 10e8;
	float t;
	for(std::vector<Anchor *>::iterator it = m_anchors.begin(); it != m_anchors.end(); ++it) {
		if((*it)->intersect(ray, t, m_hitTolerance)) {
			if(t < minDist) {
				m_activeAnchor = (*it);
				minDist = t;
				hit = ray.travel(t);
			}
		}
	}
	return minDist < 10e8;
}

void AnchorGroup::moveAnchor(Vector3F & dis)
{
	if(!m_activeAnchor) return;
	m_activeAnchor->translate(dis);
}

unsigned AnchorGroup::numAnchors() const
{
	return (unsigned)m_anchors.size();
}

Anchor * AnchorGroup::firstAnchor()
{
	m_anchorIt = m_anchors.begin();
	return *m_anchorIt;
}

Anchor * AnchorGroup::nextAnchor()
{
	m_anchorIt++;
	if(!hasAnchor()) return 0;
	return *m_anchorIt; 
}

bool AnchorGroup::hasAnchor()
{
	return m_anchorIt != m_anchors.end();
}

void AnchorGroup::setHitTolerance(float val)
{
	m_hitTolerance = val;
}

std::vector<Anchor *> & AnchorGroup::data()
{
	return m_anchors;
}
//:~
