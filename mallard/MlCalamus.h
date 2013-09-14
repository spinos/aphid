/*
 *  MlCalamus.h
 *  mallard
 *
 *  Created by jian zhang on 9/14/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <MlFeather.h>
class MlCalamus
{
public:
	MlCalamus();
	void bindToFace(unsigned faceIdx, float u, float v);
	virtual ~MlCalamus();
	
	unsigned faceIdx() const;
	float patchU() const;
	float patchV() const;
private:
	unsigned m_faceIdx;
	float m_patchU, m_patchV, m_rotX, m_rotY, m_scale, padding;
	MlFeather * m_geo;
};