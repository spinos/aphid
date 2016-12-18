/*
 *  RayMarch.h
 *  btree
 *
 *  Created by jian zhang on 5/7/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <deque>
#include <math/Ray.h>
#include <math/BoundingBox.h>

namespace aphid {

class RayMarch
{
public:
	RayMarch();
	void initialize(const BoundingBox & bb, const float & gridSize);
	bool begin(const Ray & r);
	bool end();
	void step();
	const BoundingBox gridBBox() const;
	const std::deque<Vector3F> touched(const float & threshold, BoundingBox & limit) const;
	const BoundingBox computeBBox(const Vector3F & p) const;

private:
	
	Ray m_path;
	BoundingBox m_limit, m_current;
	float m_gridSize;
};

}