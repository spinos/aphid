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
	Anchor * firstAnchor();
	Anchor * nextAnchor();
	bool hasAnchor();
	void setHitTolerance(float val);
private:
	std::vector<Anchor *> m_anchors;
	std::vector<Anchor *>::iterator m_anchorIt;
	Anchor * m_activeAnchor;
	float m_hitTolerance;
};