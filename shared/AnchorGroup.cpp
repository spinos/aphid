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
	m_activeAnchorIdx = 0;
	m_activeAnchor = 0;
	float minDist = 10e8;
	float t;
	int i = 0;
	for(std::vector<Anchor *>::iterator it = m_anchors.begin(); it != m_anchors.end(); ++it) {
		if((*it)->intersect(ray, t, m_hitTolerance)) {
			if(t < minDist) {
				m_activeAnchor = (*it);
				m_activeAnchorIdx = i;
				minDist = t;
				hit = ray.travel(t);
			}
		}
		i++;
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

float AnchorGroup::getHitTolerance() const
{
	return m_hitTolerance;
}

std::vector<Anchor *> & AnchorGroup::data()
{
	return m_anchors;
}

bool AnchorGroup::activeAnchor(unsigned & idx) const
{
	if(!m_activeAnchor) return false;
	idx = m_activeAnchorIdx;
	return true;
}

void AnchorGroup::removeLast()
{
	if(numAnchors() < 1) return;
	m_anchorIt = m_anchors.begin();
	m_anchorIt += numAnchors() - 1;
	m_anchors.erase(m_anchorIt);
}
//:~
