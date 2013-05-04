/*
 *  AnchorGroup.h
 *  masq
 *
 *  Created by jian zhang on 4/15/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <Anchor.h>
#include <vector>
#include <AllMath.h>

class AnchorGroup {
public:
	AnchorGroup();
	virtual ~AnchorGroup();
	void reset();
	void addAnchor(Anchor * a);
	bool pickupAnchor(const Ray & ray, Vector3F & hit);
	void moveAnchor(Vector3F & dis);
	unsigned numAnchors() const;
	unsigned numAnchorPoints();
	Anchor * firstAnchor();
	Anchor * nextAnchor();
	bool hasAnchor();
	
	std::vector<Anchor *> & data();
	bool activeAnchorIdx(unsigned & idx) const;
	Anchor * activeAnchor() const;
	void removeLast();
	void removeActive();
	
	void setHitTolerance(float val);
	float getHitTolerance() const;
private:
	std::vector<Anchor *> m_anchors;
	std::vector<Anchor *>::iterator m_anchorIt;
	Anchor * m_activeAnchor;
	unsigned m_activeAnchorIdx;
	float m_hitTolerance;
};