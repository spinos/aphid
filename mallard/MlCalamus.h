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
	
	unsigned faceIdx() const;
	float patchU() const;
	float patchV() const;
	float rotateX() const;
	float rotateY() const;
	float scale() const;
	
	void setRotate(const float& x, const float y);
	void setScale(const float & x);
private:
	MlFeather * m_geo;
	unsigned m_faceIdx;
	float m_patchU, m_patchV, m_rotX, m_rotY, m_scale, padding;
};