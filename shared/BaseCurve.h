/*
 *  BaseCurve.h
 *  fit
 *
 *  Created by jian zhang on 4/22/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <AllMath.h>
#include <vector>
#include <Geometry.h>

namespace aphid {

class BoundingBox;
class BaseCurve : public Geometry {
public:
	BaseCurve();
	virtual ~BaseCurve();
	
// overrid typed
	virtual const Type type() const;
	
	void cleanup();
	
	void createVertices(unsigned num);
	
	const unsigned & numVertices() const;
	const unsigned numSegments() const;
	void computeKnots();
	
	unsigned segmentByParameter(float param) const;
	unsigned segmentByLength(float param) const;
	
	const Vector3F & getCv(unsigned idx) const;
	const float & getKnot(unsigned idx) const;
	
	void fitInto(BaseCurve & another);
	
	virtual Vector3F interpolate(float param) const;
	virtual Vector3F interpolate(float param, Vector3F * data) const;
	virtual float length() const;
	
	Vector3F calculateStraightPoint(float t, unsigned k0, unsigned k1, Vector3F * data) const;
	
	void findNeighborKnots(float param, unsigned & nei0, unsigned & nei1) const;
	
	const float * cvV() const;
	
	Vector3F * m_cvs;
	float * m_knots;
	unsigned m_numVertices;
	float m_hullLength;

// override geometry
	virtual const unsigned numComponents() const;
	virtual const BoundingBox calculateBBox() const;
	virtual const BoundingBox calculateBBox(unsigned icomponent) const;
	virtual bool intersectRay(const Ray * r);
	virtual bool intersectRay(unsigned icomponent, const Ray * r);
	
	static float RayIntersectionTolerance;
private:
};

}