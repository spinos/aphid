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
		void reset(const Vector3F & n, const Vector3F & p, const Vector3F & f, const float & r) {
			m_plane = Plane(n, p);
			m_origin = p;
			m_ellipseCenter = p + f * r * 0.5f;
			m_front = f.normal();
			m_maxRadius = r * 0.5f;
			m_maxAngle = 0.f;
		}
		
		Plane m_plane;
		Vector3F m_origin, m_front, m_ellipseCenter;
		float m_maxAngle, m_currentAngle, m_componentMaxAngle, m_frontFacingThreshold, m_maxRadius;
	};
	
	Patch();
	Patch(const Vector3F & p0, const Vector3F & p1, const Vector3F & p2, const Vector3F & p3);
	virtual ~Patch();
	
	float planarDistanceTo(const Vector3F & po, Vector3F & closestP) const;
	bool pushPlane(PushPlaneContext * ctx) const;
	
	Vector3F vertex(int idx) const;
	Vector3F center() const;
	Matrix33F tangentFrame() const;
	
private:
	Segment m_segs[4];
};