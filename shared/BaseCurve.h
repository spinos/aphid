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
	
	void addVertex(const Vector3F & vert);
	unsigned numVertices() const;
	void computeKnots();
	
	Vector3F getVertex(unsigned idx) const;
	float getKnot(unsigned idx) const;
	
	void fitInto(BaseCurve & another);
	Vector3F interplate(float param) const;
private:
	std::vector<Vector3F> m_vertices;
	std::vector<float> m_knots;
	float m_length;
};