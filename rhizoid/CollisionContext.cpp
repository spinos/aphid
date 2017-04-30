/*
 *  CollisionContext.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 3/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "CollisionContext.h"

namespace aphid {

bool CollisionContext::contact(const Vector3F & q,
			const float & r) 
{
	return _pos.distanceTo(q) < (_radius + _minDistance + r);
}

BoundingBox CollisionContext::getBBox() const
{
	BoundingBox b;
	b.set(_pos, _radius);
	return b;
}

}