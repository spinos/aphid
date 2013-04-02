/*
 *  RayIntersectionContext.h
 *  lapl
 *
 *  Created by jian zhang on 3/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <BoundingBox.h>
#include <Primitive.h>
#include <Geometry.h>
class RayIntersectionContext {
public:
	enum ComponentType {
		TFace,
		TVertex
	};
	RayIntersectionContext();
	virtual ~RayIntersectionContext();
	
	void reset();
	void setBBox(const BoundingBox & bbox);
	BoundingBox getBBox() const;
	
	void setComponentFilerType(ComponentType ft);
	
	void verbose() const;

	BoundingBox m_bbox;
	Vector3F m_hitP, m_hitN;
	float m_minHitDistance;
	int m_level;
	Geometry * m_geometry;
	unsigned m_componentIdx;
	Primitive * m_primitive;
	char m_success;
	char * m_cell;
	ComponentType m_componentType;
private:
};