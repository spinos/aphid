/*
 *  SplitEvent.h
 *  kdtree
 *
 *  Created by jian zhang on 10/22/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <BoundingBox.h>

namespace aphid {

class BuildKdTreeContext;

class SplitEvent {
public:
	SplitEvent();
	
	void setEmpty();
	/// split box into left and right box
	void setBAP(const BoundingBox & b, int a, float p);
	
	const float & getPos() const;
	const int & getAxis() const;
	
	void setLeftRightNumPrim(const unsigned &leftNumPrim, const unsigned &rightNumPrim);
	
	const float & getCost() const;
	
	const int & leftCount() const;
	const int & rightCount() const;
	
	void calculateTightBBoxes(const BoundingBox &box, BoundingBox &leftBBox, BoundingBox &rightBBox);
	void calculateCost(const float & a, const float & vol);
	float area() const;
	int side(const BoundingBox &box) const;
	void updateLeftBox(const BoundingBox &box);
	void updateRightBox(const BoundingBox &box);
	const BoundingBox & leftBound() const;
	const BoundingBox & rightBound() const;
	bool hasBothSides() const;
	const char & isEmpty() const;
	void verbose() const;
	
private:
	BoundingBox m_leftBox, m_rightBox, m_lftBound, m_rgtBound;
	float m_pos;
	int m_axis;
	float m_cost;
	int m_leftNumPrim, m_rightNumPrim;
	char m_isEmpty;
};

}
