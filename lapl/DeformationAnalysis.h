/*
 *  DeformationAnalysis.h
 *  lapl
 *
 *  Created by jian zhang on 3/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "BaseMesh.h"
class Matrix33F;
class DeformationAnalysis {
public:
	DeformationAnalysis();
	virtual ~DeformationAnalysis();
	
	void setMeshes(BaseMesh * a, BaseMesh * b);
	void computeR();
	
	BaseMesh * getMeshA() const;
	BaseMesh * getMeshB() const;
	Matrix33F getR(unsigned idx) const;
	Vector3F getT(unsigned idx) const;
	float getS(unsigned idx) const;
	
	unsigned numVertices() const;
	Vector3F restP(unsigned idx) const;
	Vector3F differential(unsigned idx) const;
	Vector3F transformedDifferential(unsigned idx) const;
private:
	void svdRotation();
	void shtRotation();
	void edgeScale();
	
	BaseMesh * m_restMesh;
	BaseMesh * m_effectMesh;
	Matrix33F * m_Ri;
	float * m_scale;
};
