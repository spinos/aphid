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
#include <BoundingBox.h>
class MlFeather;
class CollisionRegion;
class MlFeatherCollection;
class MlCalamus
{
public:
	MlCalamus();
	void bindToFace(unsigned faceIdx, float u, float v);
	
	void bendFeather();
	void bendFeather(const Vector3F & origin, const Matrix33F& space);
	void curlFeather();
	void computeFeatherWorldP(const Vector3F & origin, const Matrix33F& space);
	
	MlFeather * feather() const;
	short featherIdx() const;
	short featherNumSegment() const;
	unsigned faceIdx() const;
	float patchU() const;
	float patchV() const;
	float rotateX() const;
	float rotateY() const;
	float scale() const;
	float realScale() const;
	unsigned bufferStart() const;
	
	void setFeatherId(unsigned x);
	void setPatchU(float u);
	void setPatchV(float v);
	void setRotateX(const float& x);
	void setRotateY(const float& y);
	void setScale(const float & x);
	void setBufferStart(unsigned x);
	
	void collideWith(CollisionRegion * skin, const BoundingBox & bbox);
	void collideWith(CollisionRegion * skin, const Vector3F & center);
	
	static MlFeatherCollection * FeatherLibrary;

private:
	unsigned m_faceIdx, m_featherId, m_bufStart;
	float m_patchU, m_patchV, m_rotX, m_rotY, m_scale;
};