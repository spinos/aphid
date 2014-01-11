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
class RenderEngine {
public:
	RenderEngine();
	virtual ~RenderEngine();
	
	virtual void setCamera(BaseCamera * camera);
	virtual void setResolution(unsigned resx, unsigned resy);
	virtual void setLights(LightGroup * lights);
	
	unsigned resolutionX() const;
	unsigned resolutionY() const;
	
	BaseCamera * camera() const;
	LightGroup * lights() const;

	virtual void preRender();
	virtual void render();
	virtual void postRender();
protected:
	
private:
	BaseCamera * m_camera;
	LightGroup * m_lights;
	unsigned m_resolutionX, m_resolutionY;
};