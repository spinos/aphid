/*
 *  RenderEngine.h
 *  aphid
 *
 *  Created by jian zhang on 12/31/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <boost/timer.hpp>
class BaseCamera;
class LightGroup;
class ShaderGroup;
class RenderOptions;
class RenderEngine {
public:
	RenderEngine();
	virtual ~RenderEngine();
	
	void setOptions(RenderOptions * options);
	void setLights(LightGroup * lights);
	void setShaders(ShaderGroup * shaders);
	
	BaseCamera * camera() const;
	LightGroup * lights() const;
	ShaderGroup * shaders() const;
	RenderOptions * options() const;

	virtual void preRender();
	virtual void render();
	virtual void postRender();
protected:
	void startTimer();
	float elapsedTime();
private:
    boost::timer m_met;
	LightGroup * m_lights;
	ShaderGroup * m_shaders;
	RenderOptions * m_options;
};