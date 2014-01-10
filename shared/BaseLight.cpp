/*
 *  BaseLight.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseLight.h"

BaseLight::BaseLight() 
{
	m_lightColor = Float3(1.f, 1.f, 1.f);
	m_intensity = 1.f;
	bbox()->setMin(-4.f, -4.f, -4.f);
	bbox()->setMax(4.f, 4.f, 4.f);
}

BaseLight::~BaseLight() {}

void BaseLight::setLightColor(float r, float g, float b)
{
	m_lightColor.x = r;
	m_lightColor.y = g;
	m_lightColor.z = b;
}

Float3 BaseLight::lightColor() const
{
	return m_lightColor;
}

void BaseLight::setIntensity(float x)
{
	m_intensity = x;
}

float BaseLight::intensity() const
{
	return m_intensity;
}