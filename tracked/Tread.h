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
	struct Section {
		enum SectionType {
			tLinear = 0,
			tAngular = 1
		};
		
		SectionType _type;
		float _initialAngle;
		float _eventualAngle;
		float _deltaAngle;
		float _rotateRadius;
		Vector3F _rotateAround;
		Vector3F _initialPosition;
		Vector3F _eventualPosition;
		Vector3F _deltaPosition;
		int _numSegments;
	};
	
	Tread();
	
	void setWidth(const float & x);
	void setThickness(const float & x);
	void begin();
	bool end();
	void next();
	const Matrix44F currentSpace() const;
	const bool currentIsShoe() const;
	const float shoeLength() const;
	const float width() const;
	const float shoeWidth() const;
	const float padWidth() const;
	const float padX() const;
	const float pinLength() const;
	const float segLength() const;
	const float shoeThickness() const;
	const float pinThickness() const;
	const float pinHingeFactor() const;
	void addSection(const Section & sect);
	void clearSections();
	void computeSections(const float & addThreshold = .5f);
	
	static const float computeShoeLength(const float & sprocketR);
	static const float computePinLength(const float & sprocketR, const float & toothW);
	static const float computeShoeWidth(const float & trackW, const float & toothW);
	static const float computePinThickness(const float & trackH);
	
	static float ShoeLengthFactor;
	static float ShoeHingeRise;
	static float ToothWidth;
	static float ToothHeight;
	static float SprocketRadius;
	static float PinWidthFactor;
private:
	struct Iterator {
		Matrix33F rot;
		Vector3F origin;
		int numShoe, numPin;
		bool isShoe;
		int currentSection;
	};
	Iterator m_it;
	float m_width, m_thickness, m_shoeLength;
	std::deque<Section> m_sections;
};