/*
 *  MlRachis.h
 *  mallard
 *
 *  Created by jian zhang on 9/20/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>

class CollisionRegion;

class MlRachis {
public:
	MlRachis();
	~MlRachis();
	
	void create(unsigned x);
	void computeLengths(float * segL, float fullL);
	void reset();
	void bend();
	void bend(unsigned faceIdx, float patchU, float patchV, const Vector3F & oriP, const Matrix33F & space, float radius, CollisionRegion * collide);
	void curl(const float & fullPitch);
	Matrix33F getSpace(short idx) const;
	float * angles() const;
private:
	char isInside(const Vector3F & t, const Vector3F & onp, const Vector3F & nor);
	float bouncing(const Vector3F & a, const Vector3F & b, const Vector3F & c);
	float distanceFactor(const Vector3F & a, const Vector3F & b, const Vector3F & c);
	float pushToSurface(const Vector3F & wv, const Matrix33F & space);
	float matchNormal(const Vector3F & wv, const Matrix33F & space);
	void moveForward(const Matrix33F & space, float distance, Vector3F & dst);
	void rotateForward(const Matrix33F & space, Matrix33F & dst);
	unsigned m_numSpace;
	Matrix33F * m_spaces;
	float * m_angles;
	float * m_lengths;
	float * m_lengthPortions;
};