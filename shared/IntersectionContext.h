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
namespace aphid {

class IntersectionContext : public PrimitiveFilter {
	
	BoundingBox m_bbox;
	
public:
	IntersectionContext();
	virtual ~IntersectionContext();
	
	void reset();
	void reset(const Ray & ray);
	void reset(const Ray & ray, 
				const Vector3F & ref, 
				const float & scaling);
	void setBBox(const BoundingBox & bbox);
	BoundingBox getBBox() const;
	
	void setNormalReference(const Vector3F & nor);
	
	void verbose() const;

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
	float m_tmin, m_tmax;
	
private:
};

class BoxIntersectContext : public BoundingBox {
	
	std::vector<int> m_prims;
	int m_cap;
	bool m_exact;
	
public:
	BoxIntersectContext();
	virtual ~BoxIntersectContext();
	
	void reset(int maxNumPrim = 1, bool beExact = false);
	void addPrim(const int & i);
	int numIntersect() const;
	bool isExact() const;
	bool isFull() const;
	
};

}