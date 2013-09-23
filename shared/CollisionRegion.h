/*
 *  CollisionRegion.h
 *  mallard
 *
 *  Created by jian zhang on 9/21/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
class IntersectionContext;

class CollisionRegion {
public:
	CollisionRegion();
	virtual ~CollisionRegion();
	
	Vector3F getClosestPoint(const Vector3F & origin);
	
	virtual void resetCollisionRegion(unsigned idx);
	virtual void resetCollisionRegionAround(unsigned idx, const Vector3F & p, const float & d);
	virtual void closestPoint(const Vector3F & origin, IntersectionContext * ctx) const;
	
	unsigned numRegionElements() const;
	unsigned regionElementIndex(unsigned idx) const;
	
	unsigned regionElementStart() const;
	void setRegionElementStart(unsigned x);
	
	std::vector<unsigned> * regionElementIndices();
private:
	std::vector<unsigned> m_regionElementIndices;
	unsigned m_regionElementStart;
	IntersectionContext * m_ctx;
};