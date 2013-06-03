/*
 *  LODQuad.h
 *  easymodel
 *
 *  Created by jian zhang on 11/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include "BaseQuad.h"

class LODQuad : public BaseQuad {
public:
	LODQuad();
	virtual ~LODQuad();

	void setDetail(float d, int i);
	void setUniformDetail(float d);
	float getDetail(int i) const;
	
	void evaluateSurfaceLOD(float u, float v, float * detail) const;
	float getMaxLOD() const;
	float getMaxEdgeLength() const;
	
	float _details[4];
};
