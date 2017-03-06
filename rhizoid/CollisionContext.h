/*
 *  CollisionContext.h
 *  proxyPaint
 *
 *  Created by jian zhang on 3/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <math/BoundingBox.h>

namespace aphid {

class CollisionContext {

public:	
		Vector3F _pos;
		int _bundleIndex;
		int _minIndex;
		float _minDistance;
		float _maxDistance;
		float _bundleScaling;
		float _radius;
		
		BoundingBox getBBox() const;
		
		bool contact(const Vector3F & q,
					const float & r);
	};
	
}