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
class CollisionRegion {
public:
	CollisionRegion();
	virtual ~CollisionRegion();
	
	virtual void resetCollisionRegion(unsigned idx);
	
	unsigned numRegionElements() const;
	unsigned regionElementIndex(unsigned idx) const;
	
	unsigned regionElementStart() const;
	void setRegionElementStart(unsigned x);
	
	std::vector<unsigned> * regionElementIndices();
private:
	std::vector<unsigned> m_regionElementIndices;
	unsigned m_regionElementStart;
};