/*
 *  LightGroup.h
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <vector>
class BaseLight;
class LightGroup {
public:
	LightGroup();
	virtual ~LightGroup();
	void addLight(BaseLight * l);
	unsigned numLights() const;
	BaseLight * getLight(unsigned idx) const;
private:
	std::vector<BaseLight *> m_lights;
};