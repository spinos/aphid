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
#include <geom/Geometry.h>
#include <Ray.h>
#include <Vector2F.h>

namespace aphid {

class IntersectionContext : public PrimitiveFilter {
	
	BoundingBox m_bbox;
	float m_splatR;
	
public:
	IntersectionContext();
	virtual ~IntersectionContext();
	
	void reset();
	void reset(const Ray & ray);
	void reset(const Ray & ray, const float & delta);
	void reset(const Ray & ray, 
				const Vector3F & ref, 
				const float & scaling);
	void reset(const Beam & beam, const float & delta);
	
	void setBBox(const BoundingBox & bbox);
	BoundingBox getBBox() const;
	
	void setSplatRadius(const float & x);
	const float splatRadius() const;
	
	void setNormalReference(const Vector3F & nor);
	
	void verbose() const;

	Ray m_ray;
	Beam m_beam;
	Vector3F m_hitP, m_hitN, m_closestP, m_refN, m_originP;
	Vector2F m_patchUV;
	float m_minHitDistance, m_elementHitDistance;
	int m_level;
	int m_leafIdx;
	Geometry * m_geometry;
	unsigned m_componentIdx, m_curComponentIdx;
	char m_success;
	char m_enableNormalRef;
	char twoSided;
	char * m_cell;
	float m_coord[3];
	float m_tmin, m_tmax, m_tdelta;
	
private:
};

class BoxIntersectContext : public BoundingBox {
	
	std::vector<int> m_prims;
	int m_cap;
	bool m_exact;
	
public:	
	BoxIntersectContext();
	BoxIntersectContext(const BoundingBox & a);
	virtual ~BoxIntersectContext();
	
	void reset(int maxNumPrim = 1, bool beExact = false);
	void addPrim(const int & i);
	int numIntersect() const;
	const std::vector<int> & primIndices() const;
	bool isExact() const;
	bool isFull() const;
	
};

}