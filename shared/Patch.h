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
#include <BoundingBox.h>

class Patch : public Plane {
public:
	struct PushPlaneContext {
		void reset(const Vector3F & n, const Vector3F & p, const Vector3F & f, const float & r) {
			m_plane = Plane(n, p);
			m_origin = p;
			m_front = f.normal();
			m_maxRadius = r;
			m_convergeThreshold = r * r / 128.f;
			m_maxAngle = 0.f;
			m_plane.getNormal(m_up); 
		}
		
		bool isConverged() {
			return (m_componentBBox.area() < m_convergeThreshold);
		}
		
		BoundingBox m_componentBBox;
		Plane m_plane;
		Vector3F m_origin, m_front, m_up;
		float m_maxAngle, m_currentAngle, m_maxRadius, m_convergeThreshold;
	};
	
	Patch();
	Patch(const Vector3F & p0, const Vector3F & p1, const Vector3F & p2, const Vector3F & p3);
	virtual ~Patch();
	
	float planarDistanceTo(const Vector3F & po, Vector3F & closestP) const;
	bool pushPlane(PushPlaneContext * ctx) const;
	bool isBehind(const Vector3F & po, Vector3F & nr) const;
	
	Vector3F vertex(int idx) const;
	Vector3F center() const;
	Matrix33F tangentFrame() const;
	Vector3F point(float u, float v) const;
	
private:
	Segment m_segs[4];
};