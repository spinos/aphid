/*
 *  RayMarch.h
 *  btree
 *
 *  Created by jian zhang on 5/7/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Ray.h>
#include <BoundingBox.h>
class RayMarch
{
public:
	RayMarch();
	void initialize(const BoundingBox & bb, const float & gridSize);
	bool begin(const Ray & r);
	bool end();
	void step();
	const BoundingBox gridBBox() const;
private:
	
	Ray m_path;
	BoundingBox m_limit, m_current;
	float m_gridSize;
};