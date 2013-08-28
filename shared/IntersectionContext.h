/*
 *  IntersectionContext.h
 *  lapl
 *
 *  Created by jian zhang on 3/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <BoundingBox.h>
#include <Primitive.h>
#include <PrimitiveFilter.h>
#include <Geometry.h>

class IntersectionContext : public PrimitiveFilter {
public:
	IntersectionContext();
	virtual ~IntersectionContext();
	
	void reset();
	void setBBox(const BoundingBox & bbox);
	BoundingBox getBBox() const;
	
	void setNormalReference(const Vector3F & nor);
	
	void verbose() const;

	BoundingBox m_bbox;
	Vector3F m_hitP, m_hitN, m_closest, m_refN;
	float m_minHitDistance;
	int m_level;
	Geometry * m_geometry;
	unsigned m_componentIdx;
	char m_success;
	char m_enableNormalRef;
	char * m_cell;
	float m_coord[3];
private:
};