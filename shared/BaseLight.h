/*
 *  BaseLight.h
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "BaseTransform.h"
class BaseLight : public BaseTransform {
public:
	BaseLight();
	virtual ~BaseLight();
	
	void setLightColor(float r, float g, float b);
	Float3 lightColor() const;
	
	void setIntensity(float x);
	float intensity() const;
protected:

private:
	Float3 m_lightColor;
	float m_intensity;
};