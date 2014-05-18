/*
 *  Tread.h
 *  tracked
 *
 *  Created by jian zhang on 5/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
class Tread {
public:	
	Tread();
	void setOrigin(const Vector3F & p);
	void setSpan(const float & x);
	void setRadius(const float & x);
	void setWidth(const float & x);
	int computeNumShoes();
	void begin();
	bool end();
	void next();
	const Matrix44F currentSpace() const;
	const bool currentIsShoe() const;
	const float shoeLength() const;
	const float width() const;
	static float ShoeThickness;
	static float PinThickness;
private:
	struct Iterator {
		Vector3F origin;
		int numShoe, numPin;
		float angle;
		bool isShoe;
	};
	Iterator m_it;
	Vector3F m_origin;
	int m_numShoes, m_numPins;
	float m_span, m_radius, m_width, m_shoeLength;
	
};