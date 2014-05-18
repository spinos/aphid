/*
 *  Tread.cpp
 *  tracked
 *
 *  Created by jian zhang on 5/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Tread.h"
float Tread::ShoeThickness = 0.5f;
float Tread::PinThickness = 0.35f;
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
	m_shoeLength = PI * m_radius / 13.f;
	m_radius = m_shoeLength * 13 / PI;
	
	const int onspan = m_span / m_shoeLength;
	m_span = m_shoeLength * onspan;
	 
	m_numShoes = onspan * 2 + 13 * 2;
	m_numPins = m_numShoes;
	return m_numShoes;
}

void Tread::begin() 
{
	m_it.origin = m_origin;
	m_it.isShoe = true;
	m_it.numShoe = 0;
	m_it.numPin = 0;
	m_it.angle = 0.f;
}

bool Tread::end() 
{
	return (m_it.numShoe == m_numShoes && m_it.numPin == m_numPins);
}

void Tread::next()
{
	if(m_it.isShoe) m_it.numShoe++;
	else m_it.numPin++;
	
	m_it.isShoe = !m_it.isShoe;
	m_it.origin += Vector3F::ZAxis * m_shoeLength * 0.5f;	
}

const Matrix44F Tread::currentSpace() const
{
	Matrix44F mat;
	mat.setTranslation(m_it.origin);
	return mat;
}

const bool Tread::currentIsShoe() const
{
	return m_it.isShoe;
}

const float Tread::shoeLength() const { return m_shoeLength; }
