/*
 *  Segment.h
 *  mallard
 *
 *  Created by jian zhang on 9/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <Ray.h>

class Segment : public Ray {
public:
	Segment();
	Segment(const Vector3F& pfrom, const Vector3F& pto);
	float distanceTo(const Vector3F & po, Vector3F & closestP) const;
private:
};