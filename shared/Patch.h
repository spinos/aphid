/*
 *  Patch.h
 *  mallard
 *
 *  Created by jian zhang on 9/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Plane.h>
#include <Segment.h>
class Patch : public Plane {
public:
	Patch();
	Patch(const Vector3F & p0, const Vector3F & p1, const Vector3F & p2, const Vector3F & p3);
	
	float planarDistanceTo(const Vector3F & po, Vector3F & closestP) const;
	virtual ~Patch();
	
	Vector3F vertex(int idx) const;
	
private:
	Segment m_segs[4];
};