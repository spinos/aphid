/*
 *  AnchorGroup.cpp
 *  masq
 *
 *  Created by jian zhang on 4/15/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "AnchorGroup.h"
#include <Ray.h>

namespace aphid {

AnchorGroup::AnchorGroup() {m_activeAnchor = 0;}
AnchorGroup::~AnchorGroup() {}

void AnchorGroup::reset()
{
	m_activeAnchor = 0;
	for(std::vector<Anchor *>::iterator it = m_anchors.begin(); it != m_anchors.end(); ++it) {
		delete *it;
	}
	m_anchors.clear();
	m_relevantIndex.clear();
}

void AnchorGroup::addAnchor(Anchor * a)
{
    a->setSize(getHitTolerance());
	m_anchors.push_back(a);
	m_relevantIndex.push_back(0);
}

bool AnchorGroup::pickupAnchor(const Ray & ray, Vector3F & hit)
{
	m_activeAnchorIdx = 0;
	m_activeAnchor = 0;
	float minDist = 10e8;
	float t;
	int i = 0;
	for(std::vector<Anchor *>::iterator it = m_anchors.begin(); it != m_anchors.end(); ++it) {
		if((*it)->intersect(ray, t, getHitTolerance())) {
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

unsigned AnchorGroup::numAnchorPoints()
{
	unsigned numPoints = 0;
	for(Anchor *a = firstAnchor(); hasAnchor(); a = nextAnchor()) {
		numPoints += a->numPoints();
	}
	return numPoints;
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

std::vector<Anchor *> & AnchorGroup::data()
{
	return m_anchors;
}

bool AnchorGroup::hasActiveAnchor() const
{
    return m_activeAnchor != 0;
}

bool AnchorGroup::activeAnchorIdx(unsigned & idx) const
{
	if(!m_activeAnchor) return false;

	idx = m_activeAnchorIdx;
	return true;
}

Anchor * AnchorGroup::activeAnchor() const
{
	return m_activeAnchor;
}

void AnchorGroup::removeLast()
{
	if(numAnchors() < 1) return;
	printf("rm last");
	removeAnchor(numAnchors() - 1);
}

void AnchorGroup::removeActive()
{
	if(!m_activeAnchor) return;
	removeAnchor(m_activeAnchorIdx);
	
	m_activeAnchor = 0;
	m_activeAnchorIdx = 0;
}

void AnchorGroup::removeAnchor(unsigned idx)
{
    printf("rm %i ", idx);
    m_anchorIt = m_anchors.begin();
    m_anchorIt += idx;
    delete *m_anchorIt;
    m_anchors.erase(m_anchorIt);
    
    std::vector<unsigned>::iterator it = m_relevantIndex.begin();
	it += idx;
	m_relevantIndex.erase(it);
}

void AnchorGroup::removeRelevantAnchor(unsigned idx)
{
    printf("rm rel %i ", idx);
    for(unsigned i = 0; i < m_relevantIndex.size(); ++i) {
        if(m_relevantIndex[i] == idx) {
            removeAnchor(i);
            popReleventIndex(idx);
        }   
    }
}

void AnchorGroup::popReleventIndex(unsigned idx)
{
    for(unsigned i = 0; i < m_relevantIndex.size(); i++) {
        if(m_relevantIndex[i] > idx)
            m_relevantIndex[i] = m_relevantIndex[i] - 1;
    }
}

void AnchorGroup::clearSelected()
{
	m_activeAnchor = 0;
}

void AnchorGroup::setHitTolerance(float val)
{
	m_hitTolerance = val;
}

float AnchorGroup::getHitTolerance() const
{
	return m_hitTolerance;
}

void AnchorGroup::setLastReleventIndex(unsigned val)
{
    m_relevantIndex.back() = val;
}

unsigned AnchorGroup::getReleventIndex(unsigned idx) const
{
    return m_relevantIndex[idx];
}

}
//:~
