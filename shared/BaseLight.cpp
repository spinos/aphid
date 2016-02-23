/*
 *  BaseLight.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseLight.h"

namespace aphid {

BaseLight::BaseLight() 
{
	m_lightColor = Float3(1.f, 1.f, 1.f);
	m_intensity = 1.f;
	m_samples = 1;
	m_castShadow = true;
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

void BaseLight::setLightColor(const Float3 c)
{
	m_lightColor = c;
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

void BaseLight::setSamples(int x)
{
	m_samples = x;
}

int BaseLight::samples() const
{
	return m_samples;
}
	
void BaseLight::setCastShadow(bool x)
{
	m_castShadow = x;
}

bool BaseLight::castShadow() const
{
	return m_castShadow;
}

}