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

class BaseCurve {
public:
	BaseCurve();
	virtual ~BaseCurve();
	
	void cleanup();
	
	void createVertices(unsigned num);
	void addVertex(const Vector3F & vert);
	void finishAddVertex();
	unsigned numVertices() const;
	unsigned numSegments() const;
	void computeKnots();
	
	unsigned segmentByParameter(float param) const;
	unsigned segmentByLength(float param) const;
	
	Vector3F getCv(unsigned idx) const;
	float getKnot(unsigned idx) const;
	
	void fitInto(BaseCurve & another);
	
	virtual Vector3F interpolate(float param) const;
	virtual Vector3F interpolate(float param, Vector3F * data) const;
	
	Vector3F calculateStraightPoint(float t, unsigned k0, unsigned k1, Vector3F * data) const;
	
	void findNeighborKnots(float param, unsigned & nei0, unsigned & nei1) const;
	
	static std::vector<Vector3F> BuilderVertices;
	Vector3F * m_cvs;
	float * m_knots;
	float m_length;
	unsigned m_numVertices;
private:
};