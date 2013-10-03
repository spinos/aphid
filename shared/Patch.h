/*
 *  Patch.h
 *  mallard
 *
 *  Created by jian zhang on 9/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <Plane.h>
#include <Segment.h>
class Patch : public Plane {
public:
	struct PushPlaneContext {
		void reset(const Vector3F & n, const Vector3F & p, const Vector3F & f) {
			m_plane = Plane(n, p);
			m_origin = p;
			m_front = f.normal();
			m_maxAngle = -3.14f;
		}
		
		Plane m_plane;
		Vector3F m_origin, m_front;
		float m_maxAngle, m_currentAngle, m_componentMaxAngle;
	};
	
	Patch();
	Patch(const Vector3F & p0, const Vector3F & p1, const Vector3F & p2, const Vector3F & p3);
	virtual ~Patch();
	
	float planarDistanceTo(const Vector3F & po, Vector3F & closestP) const;
	bool pushPlane(PushPlaneContext * ctx) const;
	
	Vector3F vertex(int idx) const;
	Matrix33F tangentFrame() const;
	
private:
	Segment m_segs[4];
};