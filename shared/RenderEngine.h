/*
 *  RenderEngine.h
 *  aphid
 *
 *  Created by jian zhang on 12/31/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
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
	
private:
	BaseCamera * m_camera;
	LightGroup * m_lights;
	RenderOptions * m_options;
};