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
#include <Ray.h>
#include <Vector2F.h>

class IntersectionContext : public PrimitiveFilter {
public:
	IntersectionContext();
	virtual ~IntersectionContext();
	
	void reset();
	void reset(const Ray & ray);
	void setBBox(const BoundingBox & bbox);
	BoundingBox getBBox() const;
	
	void setNormalReference(const Vector3F & nor);
	
	void verbose() const;

	BoundingBox m_bbox;
	Ray m_ray;
	Vector3F m_hitP, m_hitN, m_closestP, m_refN, m_originP;
	Vector2F m_patchUV;
	float m_minHitDistance, m_elementHitDistance;
	int m_level;
	Geometry * m_geometry;
	unsigned m_componentIdx, m_curComponentIdx;
	char m_success;
	char m_enableNormalRef;
	char twoSided;
	char * m_cell;
	float m_coord[3];
private:
};