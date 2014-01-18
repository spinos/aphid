/*
 *  LightGroup.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "LightGroup.h"
#include "DistantLight.h"
#include "PointLight.h"
LightGroup::LightGroup() 
{
	m_activeLight = 0;
}

LightGroup::~LightGroup()
{
	clearLights();
}

void LightGroup::addLight(BaseLight * l)
{
	m_lights.push_back(l);
}

unsigned LightGroup::numLights() const
{
	return m_lights.size();
}

BaseLight * LightGroup::getLight(unsigned idx) const
{
	return m_lights[idx];
}

BaseLight * LightGroup::getLight(const std::string & name) const
{
	std::vector<BaseLight *>::const_iterator it = m_lights.begin();
	for(; it != m_lights.end(); ++it) {
		if((*it)->name() == name)
			return *it;
	}
	return NULL;
}

void LightGroup::clearLights()
{
	std::vector<BaseLight *>::iterator it = m_lights.begin();
	for(; it != m_lights.end(); ++it) {
		delete *it;
	}
	m_lights.clear();
}

void LightGroup::defaultLighting()
{
	std::clog<<"default lighting\n";
	DistantLight * keyLight = new DistantLight;
	keyLight->setName("key_distant");
	Vector3F t(-10.f, 10.f, 30.f);
	keyLight->translate(t);
	Vector3F r(-.5f, -.3f, 0.f);
	keyLight->rotate(r);
	
	m_lights.push_back(keyLight);
	
	DistantLight * backLight = new DistantLight;
	backLight->setName("back_distant");
	t.set(0.f, 10.f, -30.f);
	backLight->translate(t);
	r.set(-.3f, -2.9f, 0.f);
	backLight->rotate(r);
	
	m_lights.push_back(backLight);
	
	PointLight * fillLight = new PointLight;
	fillLight->setName("fill_point");
	t.set(10.f, -4.f, 7.f);
	fillLight->translate(t);
	
	m_lights.push_back(fillLight);
}

char LightGroup::selectLight(const Ray & incident)
{
	m_activeLight = 0;
	std::vector<BaseLight *>::iterator it = m_lights.begin();
	for(; it != m_lights.end(); it++) {
		if((*it)->intersect(incident)) {
			m_activeLight = *it;
			return 1;
		}
	}
	return 0;
}

BaseLight * LightGroup::selectedLight() const
{
	return m_activeLight;
}
