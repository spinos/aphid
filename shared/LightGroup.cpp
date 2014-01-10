/*
 *  LightGroup.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "LightGroup.h"
#include "BaseLight.h"
LightGroup::LightGroup() {}
LightGroup::~LightGroup()
{
	std::vector<BaseLight *>::iterator it = m_lights.begin();
	for(; it != m_lights.end(); ++it) {
		delete *it;
	}
	m_lights.clear();
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
