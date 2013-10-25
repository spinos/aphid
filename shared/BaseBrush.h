/*
 *  BaseBrush.h
 *  mallard
 *
 *  Created by jian zhang on 9/15/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <Ray.h>
class BaseBrush {
public:
	BaseBrush();
	virtual ~BaseBrush();
	void setSpace(const Vector3F & point, const Vector3F & facing);
	void setRadius(float x);
	void setPitch(float x);
	void setMaxToeFactor(float x);
	void setNumDarts(int x);
	void resetToe();
	void setToeByIntersect(const Ray * r, bool useNormal = true);
	
	Matrix44F getSpace() const;
	float getRadius() const;
	float getPitch() const;
	float minDartDistance() const;
	const Vector3F heelPosition() const;
	const Vector3F toePosition() const;
	const Vector3F normal() const;
	const Vector3F toeDisplacement() const;
	const Vector3F toeDisplacementDelta();
	const float length() const;
	void getDartPoint(unsigned idx, Vector3F & p) const;
	
private:
	char ignoreTooClose(Vector3F p, Vector3F *data, unsigned count, float d) const;
private:
	Matrix44F m_space;
	Vector3F m_toeWorldPos, m_previousToeWorldP;
	float m_radius, m_pitch, m_maxToeFactor;
	unsigned m_numDarts;
};