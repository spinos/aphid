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
#include <Ray.h>
class BaseLight;
class LightGroup {
public:
	LightGroup();
	virtual ~LightGroup();
	void addLight(BaseLight * l);
	unsigned numLights() const;
	BaseLight * getLight(unsigned idx) const;
	BaseLight * getLight(const std::string & name) const;
	void clearLights();
	virtual char selectLight(const Ray & incident);
	BaseLight * selectedLight() const;
protected:
	virtual void defaultLighting();
private:
	std::vector<BaseLight *> m_lights;
	BaseLight * m_activeLight;
};