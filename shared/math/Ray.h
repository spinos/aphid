/*
 *  Ray.h
 *  lapl
 *
 *  Created by jian zhang on 3/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_RAY_H
#define APH_RAY_H

#include <math/Vector3F.h>

namespace aphid {

class Ray {
public:
	Ray();
	Ray(const Vector3F& pfrom, const Vector3F& vdir, float tmin, float tmax);
	Ray(const Vector3F& pfrom, const Vector3F& pto);
	void operator=(const Ray& b);
	Vector3F travel(const float & t) const;
	const Vector3F & origin() const;
	Vector3F destination() const;
	float length() const;
	const Vector3F closestPointOnRay(const Vector3F & p, float * t = NULL) const;
	Vector3F m_origin;
	float m_tmin;
	Vector3F m_dir;
	float m_tmax;
	
	const std::string str() const;
	
	friend std::ostream& operator<<(std::ostream &output, const Ray & p) {
        output << p.str();
        return output;
    }
	
/// project q on line, return t
/// if return -1 out of range
	float projectP(const Vector3F & q, Vector3F & pproj) const;
	
	float distanceToPoint(const Vector3F & q) const;
	
};

/// along ray
/// rmin at tmin rmax at tmax
class Beam {
	
	Ray m_ray;
	float m_rmin, m_rmax, drdt;
	
public:
	Beam();
	Beam(const Vector3F& pfrom, const Vector3F& pto,
		const float & rmin, const float & rmax);
	Beam(const Ray & r, const float & rmin, const float & rmax);
	
	const Ray & ray() const;
	const Vector3F & origin() const;
	Vector3F destination() const;
	float radiusAt(const float & t) const;
	void setLength(const float & tmin, const float & tmax);
	void setTmin(const float & x);
	void setTmax(const float & x);
	float length() const;
	const float & tmin() const;
	const float & tmax() const;
	float projectP(const Vector3F & q, Vector3F & pproj) const;
	
};

}
#endif