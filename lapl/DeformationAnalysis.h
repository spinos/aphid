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
private:
	BaseMesh * m_restMesh;
	BaseMesh * m_effectMesh;
	float * m_sht;
};
