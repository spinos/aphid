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
	void setStrength(float x);
	void setMaxToeFactor(float x);
	void setNumDarts(int x);
	void resetToe();
	void setToeByIntersect(const Ray * r, bool useNormal = true);
	void setTwoSided(bool b);
	void setFilterByColor(bool b);
	void setDropoff(float x);
	
	Matrix44F getSpace() const;
	float getRadius() const;
	float getPitch() const;
	float strength() const;
	float minDartDistance() const;
	const Vector3F heelPosition() const;
	const Vector3F toePosition() const;
	const Vector3F normal() const;
	const Vector3F toeDisplacement() const;
	const Vector3F toeDisplacementDelta();
	const float length() const;
	void getDartPoint(unsigned idx, Vector3F & p) const;
	bool twoSided() const;
	bool filterByColor() const;
	
	const Float3 & color() const;
	void setColor(const Float3 & c);
	
	const float & radius() const;
	const float & dropoff() const;
	
private:
	char ignoreTooClose(Vector3F p, Vector3F *data, unsigned count, float d) const;
private:
	Matrix44F m_space;
	Vector3F m_toeWorldPos, m_previousToeWorldP;
	Float3 m_color;
	float m_radius, m_pitch, m_maxToeFactor, m_strength, m_dropoff;
	unsigned m_numDarts;
	bool m_twoSided, m_filterByColor;
};