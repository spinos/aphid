/*
 *  SelectionContext.h
 *  aphid
 *
 *  Created by jian zhang on 1/22/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <BoundingBox.h>
#include <deque>
class SelectionContext {
public:
	SelectionContext();
	virtual ~SelectionContext();
	
	void reset();
	void reset(const Vector3F & center, const float & radius);
	void finish();
	
	void setDirection(const Vector3F & d);
	
	void setCenter(const Vector3F & center);
	const Vector3F & center() const;
	
	void setRadius(const float & radius);
	float radius() const;
	
	void setBBox(const BoundingBox &bbox);
	const BoundingBox & getBBox() const;
	
	char closeTo(const BoundingBox & b) const;
	char closeTo(const Vector3F & v) const;
	void addToSelection(const unsigned idx);
	
	unsigned numSelected() const;
	const std::deque<unsigned> & selectedQue() const;
	
	void verbose() const;
private:
	std::deque<unsigned> m_indices;
	BoundingBox m_bbox;
	Vector3F m_center, m_normal;
	float m_radius;
	char m_enableDirection;
};