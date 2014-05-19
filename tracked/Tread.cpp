/*
 *  Tread.cpp
 *  tracked
 *
 *  Created by jian zhang on 5/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Tread.h"
float Tread::ShoeThickness = 0.3f;
float Tread::PinThickness = 0.2f;
float Tread::ShoeWidthFactor = 0.9f;
float Tread::ShoeLengthFactor = 0.85f;
float Tread::PinToShoeLengthRatio = 0.6f;
float Tread::PinHingeFactor = 0.57f;
float Tread::ShoeHingeFactor = 0.68f;
float Tread::ShoeHingeRise = 0.27f;
float Tread::ToothWidth = .9f;
float Tread::ToothHeight = 1.f;
Tread::Tread() 
{
	m_span = 80.f;
	m_radius = 8.f;
	m_width = 16.f;
	m_origin.setZero();
}

void Tread::setOrigin(const Vector3F & p) { m_origin = p; }
void Tread::setSpan(const float & x) { m_span = x; }
void Tread::setRadius(const float & x) { m_radius = x; }
void Tread::setWidth(const float & x) { m_width = x; }

const float Tread::width() const { return m_width; }

int Tread::computeNumShoes()
{
	m_shoeLength = PI * m_radius / 7.f;
	m_radius = m_shoeLength * 7 / PI;
	
	m_numOnSpan = m_span / m_shoeLength + 1;
	m_span = m_shoeLength * m_numOnSpan;
	 
	m_numShoes = m_numOnSpan * 2 + 7 * 2;
	m_numPins = m_numShoes;
	return m_numShoes;
}

void Tread::begin() 
{
	m_it.origin = m_origin;
	m_it.isShoe = true;
	m_it.isOnSpan = true;
	m_it.numShoe = 0;
	m_it.numPin = 0;
	m_it.numOnSpan = 0;
	m_it.numOnWheel= 0;
	m_it.angle = 0.f;
	m_it.rot.setIdentity();
	m_it.spanTranslateDirection = 1.f;
}

bool Tread::end() 
{
	return (m_it.numShoe == m_numShoes && m_it.numPin == m_numPins);
}

void Tread::next()
{
	if(m_it.isOnSpan) m_it.origin += Vector3F::ZAxis * m_shoeLength * 0.5f * m_it.spanTranslateDirection;
	else m_it.rot.rotateX(-PI / 7.f * .5f);
	
	if(m_it.isShoe) m_it.numShoe++;
	else m_it.numPin++;
	
	if(m_it.isOnSpan) {
		if(m_it.isShoe) m_it.numOnSpan++;
		if(m_it.numOnSpan == m_numOnSpan) {
			m_it.isOnSpan = false;
			m_it.numOnWheel = 0;
		}
	}
	else {
		if(m_it.isShoe) m_it.numOnWheel++;
		if(m_it.numOnWheel == 7) {
			m_it.isOnSpan = true;
			m_it.numOnSpan = 0;
			m_it.spanTranslateDirection *= -1.f;
		}
	}
	
	m_it.isShoe = !m_it.isShoe;
}

const Matrix44F Tread::currentSpace() const
{
	Matrix44F mat;
	mat.setRotation(m_it.rot);
	mat.setTranslation(m_it.origin);
	Matrix44F obj;
	obj.setTranslation(0.f, -m_radius, 0.f);
	obj *= mat;
	return obj;
}

const bool Tread::currentIsShoe() const
{
	return m_it.isShoe;
}

const float Tread::shoeLength() const { return m_shoeLength; }
