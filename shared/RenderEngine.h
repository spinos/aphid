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
class RenderOptions;
class RenderEngine {
public:
	RenderEngine();
	virtual ~RenderEngine();
	
	void setCamera(BaseCamera * camera);
	void setOptions(RenderOptions * options);
	void setLights(LightGroup * lights);
	
	BaseCamera * camera() const;
	LightGroup * lights() const;
	RenderOptions * options() const;

	virtual void preRender();
	virtual void render();
	virtual void postRender();
protected:
	void startTimer();
	float elapsedTime();
private:
    boost::timer m_met;
	BaseCamera * m_camera;
	LightGroup * m_lights;
	RenderOptions * m_options;
};