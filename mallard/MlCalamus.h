/*
 *  MlCalamus.h
 *  mallard
 *
 *  Created by jian zhang on 9/14/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <AllMath.h>
class MlFeather;
class CollisionRegion;
class MlCalamus
{
public:
	MlCalamus();
	void bindToFace(unsigned faceIdx, float u, float v);
	void computeFeatherWorldP(const Vector3F & origin, const Matrix33F& space);
	
	MlFeather * feather() const;
	unsigned faceIdx() const;
	float patchU() const;
	float patchV() const;
	float rotateX() const;
	float rotateY() const;
	float scale() const;
	float realScale() const;
	unsigned bufferStart() const;
	
	void setFeather(MlFeather * geo);
	void setRotateX(const float& x);
	void setRotateY(const float& y);
	void setScale(const float & x);
	void setBufferStart(unsigned x);
	
	void collideWith(CollisionRegion * skin);
private:
	MlFeather * m_geo;
	unsigned m_faceIdx, m_bufStart;
	float m_patchU, m_patchV, m_rotX, m_rotY, m_scale;
};