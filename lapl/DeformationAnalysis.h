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

class DeformationAnalysis {
public:
	DeformationAnalysis();
	virtual ~DeformationAnalysis();
	
	void setMeshes(BaseMesh * a, BaseMesh * b);
	BaseMesh * getMeshA() const;
	BaseMesh * getMeshB() const;
	
	void computeR();
	
	unsigned numVertices() const;
	Vector3F restP(unsigned idx) const;
	Vector3F differential(unsigned idx) const;
	Vector3F transformedDifferential(unsigned idx) const;
private:
	BaseMesh * m_restMesh;
	BaseMesh * m_effectMesh;
	float * m_sht;
};
