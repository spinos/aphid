/*
 *  DeformationTarget.h
 *  lapl
 *
 *  Created by jian zhang on 3/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "BaseMesh.h"
#include <LinearMath.h>
#include <map>
class Matrix33F;
class HarmonicCoord;
class DeformationTarget {
public:
	DeformationTarget();
	virtual ~DeformationTarget();
	
	void setMeshes(BaseMesh * a, BaseMesh * b);
	void setWeightMap(HarmonicCoord * hc, unsigned valueId);
	void computeR();
	
	BaseMesh * getMeshA() const;
	BaseMesh * getMeshB() const;
	Matrix33F getR(unsigned idx) const;
	Vector3F getT(unsigned idx) const;
	float getS(unsigned idx) const;
	float getConstrainWeight(unsigned idx) const;
	
	unsigned numVertices() const;
	Vector3F restP(unsigned idx) const;
	Vector3F differential(unsigned idx) const;
	Vector3F transformedDifferential(unsigned idx) const;
	float minDisplacement() const;
	
	void genNonZeroIndices(std::map<unsigned, char > & dst) const;
	
private:
	void svdRotation();
	void edgeScale();
	void computeTRange();
	
	BaseMesh * m_restMesh;
	BaseMesh * m_effectMesh;
	HarmonicCoord * m_weightMap;
	Matrix33F * m_Ri;
	float * m_scale;
	float m_minDisplacement, m_maxDisplacement;
	unsigned m_valueId;
};
