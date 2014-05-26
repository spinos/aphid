/*
 *  Tread.cpp
 *  tracked
 *
 *  Created by jian zhang on 5/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Tread.h"

float Tread::ShoeLengthFactor = 0.86f;
float Tread::PinLengthFactor = 0.68f;
float Tread::ShoeHingeRise = 0.4f;
float Tread::ToothWidth = .8f;
float Tread::ToothHeight = 1.5f;
float Tread::SprocketRadius = 4.f;

Tread::Tread() 
{
	m_width = 8.f;
	m_thickness = 1.f;
}

void Tread::setWidth(const float & x) { m_width = x; }
void Tread::setThickness(const float & x) { m_thickness = x; } 

const float Tread::width() const { return m_width; }
const float Tread::shoeWidth() const { return m_width - ToothWidth * 2.05f; }
const float Tread::pinLength() const { return m_shoeLength - ToothWidth * 1.001f; }

const float Tread::pinHingeFactor() const
{
	return (pinLength() - pinThickness()) / segLength();
}

void Tread::begin() 
{
	m_it.currentSection = 0;
	m_it.origin = m_sections[0]._initialPosition;
	m_it.isShoe = true;
	//m_it.isOnSpan = true;
	m_it.numShoe = 0;
	m_it.numPin = 0;
	//m_it.numOnSpan = 0;
	//m_it.numOnWheel= 0;
	//m_it.angle = 0.f;
	m_it.rot.setIdentity();
	m_it.rot.rotateX(m_sections[0]._initialAngle);
	//m_it.spanTranslateDirection = 1.f;
}

bool Tread::end() 
{
	return m_it.currentSection >= (int)m_sections.size();
}

void Tread::next()
{
	const Section sect = m_sections[m_it.currentSection];
	if(sect._type == Section::tLinear) {
		m_it.origin += sect._deltaPosition * 0.5f;
	}
	else {
		m_it.rot.rotateX(sect._deltaAngle * 0.5f);
	}
	
	if(m_it.isShoe) m_it.numShoe++;
	else m_it.numPin++;
	
	m_it.isShoe = !m_it.isShoe;
	
	if(m_it.numPin >= sect._numSegments) {
		m_it.currentSection++;
		if(end()) return;
		m_it.numShoe = 0;
		m_it.numPin = 0;
		m_it.origin = m_sections[m_it.currentSection]._initialPosition;
		m_it.rot.rotateX(m_sections[m_it.currentSection]._initialAngle);
	}
}

const Matrix44F Tread::currentSpace() const
{
	Matrix44F mat;
	Matrix44F obj;
	mat.setRotation(m_it.rot);
	const Section sect = m_sections[m_it.currentSection];
	if(sect._type == Section::tLinear) {
		mat.setTranslation(m_it.origin);
		if(m_it.isShoe)
			obj.setTranslation(0.f, -0.5f * m_thickness, 0.f);
		else
			obj.setTranslation(0.f, -0.5f * m_thickness + 0.5f * m_thickness * ShoeHingeRise, 0.f);
	}
	else {
		mat.setTranslation(sect._rotateAround);
		if(m_it.isShoe)
			obj.setTranslation(0.f, - sect._rotateRadius - m_thickness * .5f, 0.f);
		else
			obj.setTranslation(0.f, - sect._rotateRadius - m_thickness * .5f +  m_thickness  * .5f * ShoeHingeRise, 0.f);
	}
	
	obj *= mat;
	return obj;
}

const bool Tread::currentIsShoe() const
{
	return m_it.isShoe;
}

const float Tread::shoeLength() const { return m_shoeLength * ShoeLengthFactor; }
const float Tread::segLength() const { return m_shoeLength; }
const float Tread::shoeThickness() const { return m_thickness; }
const float Tread::pinThickness() const { return m_thickness * .4f; }

void Tread::addSection(const Section & sect) { m_sections.push_back(sect); }
void Tread::clearSections() { m_sections.clear(); }

void Tread::computeSections()
{
	m_shoeLength = 2.f * PI * (SprocketRadius) / 11.f;
	
	std::deque<Section>::iterator it = m_sections.begin();
	for(; it != m_sections.end(); ++it) {
		Section & sect = *it;
		if(sect._type == Section::tLinear) {
			const Vector3F dp = sect._eventualPosition - sect._initialPosition;
			const float fn = dp.length() / m_shoeLength;
			sect._numSegments = fn;
			//sect._numSegments++;
			sect._deltaPosition = dp.normal() * m_shoeLength;
		}
		else {
			float da = sect._eventualAngle - sect._initialAngle;
			sect._numSegments = da * sect._rotateRadius / m_shoeLength;
			if(sect._numSegments < 0) sect._numSegments = -sect._numSegments;
			if(sect._numSegments < 1) sect._numSegments = 1;
			sect._deltaAngle = da / sect._numSegments;
		}
	}
	
	it = m_sections.begin();
	for(; it != m_sections.end(); ++it) std::cout<<" nseg "<<(*it)._numSegments;
}
